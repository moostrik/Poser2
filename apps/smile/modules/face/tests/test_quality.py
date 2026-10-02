"""Tests for the face checks: pose from the 5 keypoints, the gate, eye openness and the labels
(synthetic points, no model)."""

import math
import unittest

import numpy as np

from modules.face.FaceDetector import Face
from modules.face.FaceFinder import crop_outside, portrait_box, portrait_crop
from modules.face.quality import LEFT_EYE, RIGHT_EYE, NOSE_NEUTRAL, REJECTED_FLOOR, FaceObservation, Pose, Rejection, distance, eye_distance, eye_openness, gate, head_pose, readout, rejection_label


def keypoints(nose_dx: float = 0.0, nose_ratio: float = NOSE_NEUTRAL, roll: float = 0.0) -> np.ndarray:
    """Eyes 100 apart, mouth 100 below them; the nose tip shifted by nose_dx (relative to the eye
    distance) and at nose_ratio between the eye line and the mouth line; all rotated by roll degrees."""
    points = np.array([[-50.0, 0.0], [50.0, 0.0], [nose_dx * 100.0, nose_ratio * 100.0], [-30.0, 100.0], [30.0, 100.0]])
    angle = math.radians(roll)
    matrix = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    return points @ matrix.T + np.array([300.0, 200.0])


def eye_contour(centre: tuple[float, float], width: float, height: float) -> list[list[float]]:
    """8 points on an ellipse (including its extremes) and two pupil points."""
    ellipse = [[centre[0] + width * 0.5 * math.cos(a), centre[1] + height * 0.5 * math.sin(a)] for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)]
    return ellipse + [list(centre), list(centre)]


def landmarks(left_height: float, right_height: float, roll: float = 0.0) -> np.ndarray:
    points = np.zeros((106, 2))
    points[list(LEFT_EYE)] = eye_contour((-40.0, 0.0), 30.0, left_height)
    points[list(RIGHT_EYE)] = eye_contour((40.0, 0.0), 30.0, right_height)
    angle = math.radians(roll)
    matrix = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    return points @ matrix.T


class TestHeadPose(unittest.TestCase):
    def test_frontal_face_has_no_angles(self):
        pose = head_pose(keypoints())
        self.assertAlmostEqual(pose.yaw, 0.0, places=5)
        self.assertAlmostEqual(pose.pitch, 0.0, places=5)
        self.assertAlmostEqual(pose.roll, 0.0, places=5)

    def test_nose_to_the_right_is_a_positive_yaw_that_grows_with_the_shift(self):
        small = head_pose(keypoints(nose_dx=0.1)).yaw
        large = head_pose(keypoints(nose_dx=0.3)).yaw
        self.assertGreater(small, 5.0)
        self.assertGreater(large, small)
        self.assertAlmostEqual(head_pose(keypoints(nose_dx=-0.3)).yaw, -large, places=5)

    def test_nose_near_the_eyes_is_looking_up(self):
        self.assertGreater(head_pose(keypoints(nose_ratio=0.35)).pitch, 10.0)
        self.assertLess(head_pose(keypoints(nose_ratio=0.8)).pitch, -10.0)

    def test_roll_is_the_eye_line_and_does_not_leak_into_yaw_or_pitch(self):
        pose = head_pose(keypoints(roll=15.0))
        self.assertAlmostEqual(pose.roll, 15.0, places=3)
        self.assertAlmostEqual(pose.yaw, 0.0, places=3)
        self.assertAlmostEqual(pose.pitch, 0.0, places=3)

    def test_the_neutral_sets_the_zero_and_the_raw_ratio_is_kept(self):
        pose = head_pose(keypoints(nose_ratio=0.48), neutral=0.48)
        self.assertAlmostEqual(pose.pitch, 0.0, places=5)
        self.assertAlmostEqual(pose.nose_ratio, 0.48, places=5)
        self.assertLess(head_pose(keypoints(nose_ratio=0.58), neutral=0.48).pitch, -10.0)
        self.assertAlmostEqual(head_pose(keypoints(nose_ratio=0.58), neutral=0.58).pitch, 0.0, places=5)

    def test_eye_order_does_not_matter(self):
        points = keypoints(nose_dx=0.2, roll=10.0)
        swapped = points.copy()
        swapped[[0, 1]] = points[[1, 0]]
        self.assertEqual(head_pose(points), head_pose(swapped))


class TestEyeDistance(unittest.TestCase):
    def test_distance_between_the_eye_keypoints_whatever_the_roll(self):
        self.assertAlmostEqual(eye_distance(keypoints()), 100.0, places=5)
        self.assertAlmostEqual(eye_distance(keypoints(roll=30.0)), 100.0, places=5)


class TestGate(unittest.TestCase):
    def check(self, score=0.9, eye_dist=0.1, outside=0.0, yaw=0.0, pitch=0.0, roll=0.0):
        return gate(score, eye_dist, outside, Pose(yaw, pitch, roll, 0.48), confidence=0.7, eye_min=0.05, max_yaw=20.0, max_pitch=25.0, max_roll=30.0)

    def test_a_good_face_passes(self):
        self.assertIsNone(self.check())

    def test_each_check_in_order(self):
        self.assertEqual(self.check(score=0.5), Rejection.LOW_SCORE)
        self.assertEqual(self.check(eye_dist=0.03), Rejection.SMALL)
        self.assertEqual(self.check(outside=0.01), Rejection.OUTSIDE)
        self.assertEqual(self.check(yaw=-25.0), Rejection.TURNED)
        self.assertEqual(self.check(pitch=30.0), Rejection.PITCHED)
        self.assertEqual(self.check(roll=-40.0), Rejection.TILTED)

    def test_the_first_failing_check_is_the_reason(self):
        self.assertEqual(self.check(eye_dist=0.03, yaw=40.0), Rejection.SMALL)
        self.assertEqual(self.check(outside=0.2, yaw=40.0), Rejection.OUTSIDE)


class TestDistance(unittest.TestCase):
    def far(self, score=0.9, eye_dist=0.1, outside=0.0, yaw=0.0, pitch=0.0, roll=0.0, eyes=None, falloff=1.0):
        return distance(score, eye_dist, outside, Pose(yaw, pitch, roll, 0.48), eyes, confidence=0.7, eye_min=0.05,
                        max_yaw=20.0, max_pitch=25.0, max_roll=30.0, eye_open=0.2, falloff=falloff)

    def test_a_good_face_is_at_zero(self):
        self.assertEqual(self.far(), 0.0)
        self.assertEqual(self.far(eyes=0.3), 0.0)
        self.assertEqual(self.far(yaw=20.0, eye_dist=0.05), 0.0)       # on a limit is still good

    def test_just_past_a_limit_is_the_floor(self):
        self.assertAlmostEqual(self.far(yaw=20.01), REJECTED_FLOOR, places=3)
        self.assertAlmostEqual(self.far(eyes=0.1999), REJECTED_FLOOR, places=3)
        self.assertAlmostEqual(self.far(outside=0.0001), REJECTED_FLOOR, places=3)

    def test_it_grows_with_the_excess_to_one(self):
        half = REJECTED_FLOOR + (1.0 - REJECTED_FLOOR) * 0.5
        self.assertAlmostEqual(self.far(yaw=-30.0), half)               # a maximum: half the limit past it
        self.assertAlmostEqual(self.far(eye_dist=0.025), half)          # a minimum: half the limit short
        self.assertAlmostEqual(self.far(yaw=40.0), 1.0)
        self.assertAlmostEqual(self.far(yaw=90.0), 1.0)
        self.assertAlmostEqual(self.far(outside=0.5), half)             # half a crop width past the edge
        self.assertAlmostEqual(self.far(outside=1.5), 1.0)
        self.assertAlmostEqual(self.far(score=0.35), half)
        self.assertAlmostEqual(self.far(eyes=0.1), half)

    def test_the_worst_check_counts(self):
        self.assertAlmostEqual(self.far(yaw=30.0, pitch=50.0), 1.0)
        self.assertAlmostEqual(self.far(yaw=30.0, roll=31.0), self.far(yaw=30.0))

    def test_falloff_sets_how_far_is_far(self):
        self.assertAlmostEqual(self.far(yaw=30.0, falloff=0.5), 1.0)
        self.assertAlmostEqual(self.far(yaw=40.0, falloff=2.0), REJECTED_FLOOR + (1.0 - REJECTED_FLOOR) * 0.5)


class TestPortraitBox(unittest.TestCase):
    def face(self, roll: float = 0.0) -> Face:
        return Face(np.array([200.0, 100.0, 400.0, 300.0]), 0.9, keypoints(roll=roll))

    def test_the_box_has_the_crop_s_centre_and_size(self):
        # face 200 high, zoom 1.5: 300 wide, at the portrait's aspect 4:5 375 high, around the face's centre
        self.assertEqual(portrait_box(self.face(), 1.5, 0.0, 896, 1120), (150.0, 12.5, 450.0, 387.5))

    def test_the_offset_shifts_up_along_the_face(self):
        x0, y0, x1, y1 = portrait_box(self.face(), 1.5, 0.1, 896, 1120)
        self.assertAlmostEqual(y0, 12.5 - 20.0)
        self.assertAlmostEqual(x0, 150.0)
        # a rolled face: the box stays upright and the same size, its centre moves along the face's own up
        x0, y0, x1, y1 = portrait_box(self.face(roll=90.0), 1.5, 0.1, 896, 1120)
        self.assertAlmostEqual(x1 - x0, 300.0)
        self.assertAlmostEqual(y1 - y0, 375.0)
        self.assertAlmostEqual((y0 + y1) * 0.5, 200.0)
        self.assertAlmostEqual(abs((x0 + x1) * 0.5 - 300.0), 20.0)

    def test_the_crop_is_cut_at_that_place(self):
        image = np.zeros((400, 600, 3), np.uint8)
        image[12:388, 150:450] = 255        # the box of the first test
        crop = portrait_crop(image, self.face(), 1.5, 0.0, 896, 1120)
        self.assertEqual(crop.shape, (1120, 896, 3))
        self.assertGreater(crop[20:-20, 20:-20].mean(), 250)


class TestCropOutside(unittest.TestCase):
    # the crop of this face at zoom 1.5 is the box (150, 12.5, 450, 387.5): 300 wide
    def outside(self, width: int = 600, height: int = 400, shift=(0.0, 0.0), roll: float = 0.0, zoom: float = 1.5, offset: float = 0.0) -> float:
        move = np.array(shift)
        face = Face(np.array([200.0, 100.0, 400.0, 300.0]) + np.tile(move, 2), 0.9, keypoints(roll=roll) + move)
        return crop_outside(face, zoom, offset, 896, 1120, width, height)

    def test_a_crop_inside_the_image_is_zero(self):
        self.assertEqual(self.outside(), 0.0)
        self.assertEqual(self.outside(shift=(-150.0, -12.5)), 0.0)      # on the edge is still inside

    def test_past_each_edge_is_the_overhang_in_crop_widths(self):
        self.assertAlmostEqual(self.outside(shift=(-180.0, 0.0)), 0.1)  # left
        self.assertAlmostEqual(self.outside(shift=(0.0, -42.5)), 0.1)   # top
        self.assertAlmostEqual(self.outside(width=420), 0.1)            # right
        self.assertAlmostEqual(self.outside(height=357.5), 0.1)         # bottom

    def test_a_face_too_close_is_outside(self):
        self.assertGreater(self.outside(zoom=2.5), 0.0)

    def test_the_offset_moves_the_crop_out(self):
        self.assertAlmostEqual(self.outside(offset=0.1), 7.5 / 300.0)

    def test_the_rotated_crop_counts(self):
        # upright the crop fits this image; rolled 20 degrees its corners pass the edges
        self.assertEqual(self.outside(width=620, height=400, shift=(10.0, 0.0)), 0.0)
        self.assertGreater(self.outside(width=620, height=400, shift=(10.0, 0.0), roll=20.0), 0.0)


class TestLabels(unittest.TestCase):
    def face(self, rejection, limit, pose=Pose(0.0, 0.0, 0.0, 0.48), eyes=None, score=0.9, eye_dist=0.06, outside=0.0):
        return FaceObservation(bbox=(0, 0, 1, 1), crop=(0, 0, 1, 1), kps=(), score=score, eye_dist=eye_dist, outside=outside, pose=pose, eyes=eyes,
                               rejection=rejection, limit=limit, distance=0.0, subject=False)

    def test_the_readout_shows_every_filtered_value(self):
        self.assertEqual(readout(self.face(None, 0.0, pose=Pose(4.4, -6.0, 2.0, 0.48), eyes=0.34)), 'conf 0.90 eyes 0.06 outside 0.00 yaw 4 pitch -6 roll 2 open 0.34')
        self.assertEqual(readout(self.face(None, 0.0)), 'conf 0.90 eyes 0.06 outside 0.00 yaw 0 pitch 0 roll 0 open -')

    def test_every_reason_names_the_check_the_value_and_the_limit(self):
        self.assertEqual(rejection_label(self.face(Rejection.LOW_SCORE, 0.7, score=0.55)), 'conf 0.55 < 0.70')
        self.assertEqual(rejection_label(self.face(Rejection.SMALL, 0.05, eye_dist=0.03)), 'eyes 0.03 < 0.05')
        self.assertEqual(rejection_label(self.face(Rejection.OUTSIDE, 0.0, outside=0.12)), 'outside 0.12 > 0.00')
        self.assertEqual(rejection_label(self.face(Rejection.TURNED, 20.0, pose=Pose(32.0, 0.0, 0.0, 0.48))), 'yaw 32 > 20')
        self.assertEqual(rejection_label(self.face(Rejection.PITCHED, 25.0, pose=Pose(0.0, -41.0, 0.0, 0.48))), 'pitch -41 > 25')
        self.assertEqual(rejection_label(self.face(Rejection.TILTED, 30.0, pose=Pose(0.0, 0.0, 35.0, 0.48))), 'roll 35 > 30')
        self.assertEqual(rejection_label(self.face(Rejection.EYES, 0.2, eyes=0.12)), 'open 0.12 < 0.20')
        self.assertEqual(rejection_label(self.face(None, 0.0)), '')


class TestEyeOpenness(unittest.TestCase):
    def test_the_less_open_eye_counts(self):
        self.assertAlmostEqual(eye_openness(landmarks(12.0, 6.0), 0.0), 0.2, places=5)
        self.assertAlmostEqual(eye_openness(landmarks(6.0, 12.0), 0.0), 0.2, places=5)

    def test_a_rolled_head_is_measured_level(self):
        self.assertAlmostEqual(eye_openness(landmarks(12.0, 12.0, roll=20.0), 20.0), 0.4, places=5)


if __name__ == '__main__':
    unittest.main()
