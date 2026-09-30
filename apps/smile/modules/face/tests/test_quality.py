"""Tests for the face checks: pose from the 5 keypoints, the gate, eye openness and the quality
(synthetic points, no model)."""

import math
import unittest

import numpy as np

from modules.face.quality import LEFT_EYE, RIGHT_EYE, NOSE_NEUTRAL, Pose, Rejection, eye_openness, gate, head_pose, quality


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

    def test_eye_order_does_not_matter(self):
        points = keypoints(nose_dx=0.2, roll=10.0)
        swapped = points.copy()
        swapped[[0, 1]] = points[[1, 0]]
        self.assertEqual(head_pose(points), head_pose(swapped))


class TestGate(unittest.TestCase):
    def check(self, score=0.9, size=0.2, yaw=0.0, pitch=0.0, roll=0.0):
        return gate(score, size, Pose(yaw, pitch, roll), confidence=0.7, size_filter=0.1, max_yaw=20.0, max_pitch=25.0, max_roll=30.0)

    def test_a_good_face_passes(self):
        self.assertIsNone(self.check())

    def test_each_check_in_order(self):
        self.assertEqual(self.check(score=0.5), Rejection.LOW_SCORE)
        self.assertEqual(self.check(size=0.05), Rejection.SMALL)
        self.assertEqual(self.check(yaw=-25.0), Rejection.TURNED)
        self.assertEqual(self.check(pitch=30.0), Rejection.PITCHED)
        self.assertEqual(self.check(roll=-40.0), Rejection.TILTED)

    def test_the_first_failing_check_is_the_reason(self):
        self.assertEqual(self.check(size=0.05, yaw=40.0), Rejection.SMALL)


class TestEyeOpenness(unittest.TestCase):
    def test_the_less_open_eye_counts(self):
        self.assertAlmostEqual(eye_openness(landmarks(12.0, 6.0), 0.0), 0.2, places=5)
        self.assertAlmostEqual(eye_openness(landmarks(6.0, 12.0), 0.0), 0.2, places=5)

    def test_a_rolled_head_is_measured_level(self):
        self.assertAlmostEqual(eye_openness(landmarks(12.0, 12.0, roll=20.0), 20.0), 0.4, places=5)


class TestQuality(unittest.TestCase):
    def test_frontal_open_sharp_is_best(self):
        best = quality(Pose(0.0, 0.0, 0.0), 0.4, 1e9, 20.0, 25.0, 0.2)
        self.assertAlmostEqual(best, 1.0, places=3)
        self.assertLess(quality(Pose(15.0, 0.0, 0.0), 0.4, 1e9, 20.0, 25.0, 0.2), best)
        self.assertLess(quality(Pose(0.0, 0.0, 0.0), 0.25, 1e9, 20.0, 25.0, 0.2), best)
        self.assertLess(quality(Pose(0.0, 0.0, 0.0), 0.4, 50.0, 20.0, 25.0, 0.2), best)

    def test_the_limit_halves_the_pose_term(self):
        self.assertAlmostEqual(quality(Pose(20.0, 0.0, 0.0), 0.4, 1e9, 20.0, 25.0, 0.2), 0.5, places=3)

    def test_unmeasured_eyes_count_half(self):
        self.assertAlmostEqual(quality(Pose(0.0, 0.0, 0.0), None, 1e9, 20.0, 25.0, 0.2), 0.5, places=3)


if __name__ == '__main__':
    unittest.main()
