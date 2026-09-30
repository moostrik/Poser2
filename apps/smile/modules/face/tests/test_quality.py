"""Tests for the face checks: pose from the 5 keypoints, the gate, eye openness and the labels
(synthetic points, no model)."""

import math
import unittest

import numpy as np

from modules.face.quality import LEFT_EYE, RIGHT_EYE, NOSE_NEUTRAL, FaceObservation, Pose, Rejection, eye_distance, eye_openness, gate, head_pose, readout, rejection_label


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
    def check(self, score=0.9, eye_dist=0.1, yaw=0.0, pitch=0.0, roll=0.0):
        return gate(score, eye_dist, Pose(yaw, pitch, roll, 0.48), confidence=0.7, eye_min=0.05, eye_max=0.25, max_yaw=20.0, max_pitch=25.0, max_roll=30.0)

    def test_a_good_face_passes(self):
        self.assertIsNone(self.check())

    def test_each_check_in_order(self):
        self.assertEqual(self.check(score=0.5), Rejection.LOW_SCORE)
        self.assertEqual(self.check(eye_dist=0.03), Rejection.SMALL)
        self.assertEqual(self.check(eye_dist=0.3), Rejection.LARGE)
        self.assertEqual(self.check(yaw=-25.0), Rejection.TURNED)
        self.assertEqual(self.check(pitch=30.0), Rejection.PITCHED)
        self.assertEqual(self.check(roll=-40.0), Rejection.TILTED)

    def test_the_first_failing_check_is_the_reason(self):
        self.assertEqual(self.check(eye_dist=0.03, yaw=40.0), Rejection.SMALL)


class TestLabels(unittest.TestCase):
    def face(self, rejection, limit, pose=Pose(0.0, 0.0, 0.0, 0.48), eyes=None, score=0.9, eye_dist=0.06):
        return FaceObservation(bbox=(0, 0, 1, 1), kps=(), score=score, eye_dist=eye_dist, pose=pose, eyes=eyes,
                               rejection=rejection, limit=limit, subject=False)

    def test_the_readout_shows_every_filtered_value(self):
        self.assertEqual(readout(self.face(None, 0.0, pose=Pose(4.4, -6.0, 2.0, 0.48), eyes=0.34)), 'conf 0.90 eyes 0.06 yaw 4 pitch -6 roll 2 open 0.34')
        self.assertEqual(readout(self.face(None, 0.0)), 'conf 0.90 eyes 0.06 yaw 0 pitch 0 roll 0 open -')

    def test_every_reason_names_the_check_the_value_and_the_limit(self):
        self.assertEqual(rejection_label(self.face(Rejection.LOW_SCORE, 0.7, score=0.55)), 'conf 0.55 < 0.70')
        self.assertEqual(rejection_label(self.face(Rejection.SMALL, 0.05, eye_dist=0.03)), 'eyes 0.03 < 0.05')
        self.assertEqual(rejection_label(self.face(Rejection.LARGE, 0.25, eye_dist=0.3)), 'eyes 0.30 > 0.25')
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
