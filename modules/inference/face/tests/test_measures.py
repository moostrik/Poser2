"""What a face's points measure: the head pose from the 5 keypoints, the eye distance and the eyes' openness from the
landmarks (synthetic points, no model)."""

import math
import unittest

import numpy as np

from .. import eye_distance, eye_openness, head_pose
from ..measures import LEFT_EYE, NOSE_NEUTRAL, RIGHT_EYE


def keypoints(nose_dx: float = 0.0, nose_ratio: float = NOSE_NEUTRAL, roll: float = 0.0) -> np.ndarray:
    """Eyes 100 apart, mouth 100 below them; the nose tip shifted by nose_dx (relative to the eye
    distance) and at nose_ratio between the eye line and the mouth line; all rotated by roll degrees."""
    points = np.array([[-50.0, 0.0], [50.0, 0.0], [nose_dx * 100.0, nose_ratio * 100.0], [-30.0, 100.0], [30.0, 100.0]])
    angle = math.radians(roll)
    matrix = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    return points @ matrix.T + np.array([300.0, 200.0])


def eye_contour(center: tuple[float, float], width: float, height: float) -> list[list[float]]:
    """8 points on an ellipse (including its extremes) and two pupil points."""
    ellipse = [[center[0] + width * 0.5 * math.cos(a), center[1] + height * 0.5 * math.sin(a)] for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)]
    return ellipse + [list(center), list(center)]


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


class TestEyeOpenness(unittest.TestCase):
    def test_the_less_open_eye_counts(self):
        self.assertAlmostEqual(eye_openness(landmarks(12.0, 6.0), 0.0), 0.2, places=5)
        self.assertAlmostEqual(eye_openness(landmarks(6.0, 12.0), 0.0), 0.2, places=5)

    def test_a_rolled_head_is_measured_level(self):
        self.assertAlmostEqual(eye_openness(landmarks(12.0, 12.0, roll=20.0), 20.0), 0.4, places=5)


if __name__ == '__main__':
    unittest.main()
