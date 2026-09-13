"""Tests for AngleExtractor (joint angles from keypoints) and AngleVelExtractor (wrapped frame-to-frame velocity)."""

import math
import unittest

import numpy as np

from modules.pose.features import AngleLandmark, Angles, AngleVelocity, PointLandmark, Points2D
from modules.pose.frame import Frame
from modules.pose.nodes import AngleExtractor, AngleExtractorSettings, AngleVelExtractor, AngleVelExtractorSettings

from ._builders import FPS, frame, points, scalar, skeleton, skeleton_coords, wrap

A = AngleLandmark


def _angles(pts: Points2D, aspect_ratio: float = 0.75) -> Angles:
    settings = AngleExtractorSettings()
    settings.aspect_ratio = aspect_ratio
    return AngleExtractor(settings).process(frame(features={Points2D: pts}))[Angles]


class AngleExtractorTest(unittest.TestCase):
    def test_full_skeleton_gives_every_angle(self) -> None:
        angles = _angles(skeleton())
        self.assertEqual(angles.valid_count, len(AngleLandmark))
        self.assertTrue(angles.validate()[0])

    def test_bending_a_joint_changes_its_angle_by_the_bend(self) -> None:
        straight = _angles(skeleton())
        for name, bend in (('left_elbow', 0.6), ('right_elbow', 0.6), ('left_knee', -0.4), ('right_knee', 1.2)):
            with self.subTest(joint=name):
                bent = _angles(skeleton(**{name: bend}))
                self.assertAlmostEqual(wrap(bent[A[name]] - straight[A[name]]), bend, places=4)

    def test_bending_a_joint_leaves_the_others(self) -> None:
        straight = _angles(skeleton())
        bent = _angles(skeleton(left_elbow=0.8, right_knee=0.5))
        for joint in AngleLandmark:
            if joint in (A.left_elbow, A.right_knee):
                continue
            with self.subTest(joint=joint.name):
                self.assertAlmostEqual(bent[joint], straight[joint], places=5)

    def test_mirror_symmetric_pose_gives_equal_left_and_right(self) -> None:
        angles = _angles(skeleton(left_elbow=0.7, right_elbow=0.7, left_knee=0.3, right_knee=0.3))
        for left, right in ((A.left_shoulder, A.right_shoulder), (A.left_elbow, A.right_elbow),
                            (A.left_hip, A.right_hip), (A.left_knee, A.right_knee)):
            with self.subTest(pair=left.name):
                self.assertAlmostEqual(angles[left], angles[right], places=5)

    def test_head_facing_camera_is_zero(self) -> None:
        self.assertAlmostEqual(_angles(skeleton())[A.head], 0.0, places=5)

    def test_aspect_ratio_correction_recovers_physical_angles(self) -> None:
        # The same physical pose seen through a 3:4 and a square crop yields the same angles once each
        # extractor is told its crop's aspect ratio.
        narrow = _angles(skeleton(0.75, left_elbow=0.9), aspect_ratio=0.75)
        square = _angles(skeleton(1.0, left_elbow=0.9), aspect_ratio=1.0)
        np.testing.assert_allclose(narrow.values, square.values, atol=1e-4)

    def test_wrong_aspect_ratio_distorts_angles(self) -> None:
        right = _angles(skeleton(0.75, left_elbow=0.9), aspect_ratio=0.75)
        wrong = _angles(skeleton(0.75, left_elbow=0.9), aspect_ratio=1.0)
        self.assertGreater(abs(wrap(right[A.left_elbow] - wrong[A.left_elbow])), 0.01)

    def test_missing_keypoint_invalidates_only_its_angles(self) -> None:
        coords = skeleton_coords()
        del coords[PointLandmark.left_wrist]
        angles = _angles(points(coords))
        self.assertTrue(math.isnan(angles[A.left_elbow]))
        self.assertEqual(angles.get_score(A.left_elbow), 0.0)
        self.assertFalse(math.isnan(angles[A.left_shoulder]))

    def test_coincident_keypoints_give_nan(self) -> None:
        coords = skeleton_coords()
        coords[PointLandmark.left_wrist] = coords[PointLandmark.left_elbow]
        angles = _angles(points(coords))
        self.assertTrue(math.isnan(angles[A.left_elbow]))
        self.assertEqual(angles.get_score(A.left_elbow), 0.0)

    def test_score_is_minimum_of_keypoint_scores(self) -> None:
        pts = points(skeleton_coords(), scores={PointLandmark.left_elbow: 0.6, PointLandmark.left_wrist: 0.8})
        angles = _angles(pts)
        self.assertAlmostEqual(angles.get_score(A.left_elbow), 0.6, places=6)
        self.assertAlmostEqual(angles.get_score(A.left_shoulder), 0.6, places=6)   # elbow is its distal point
        self.assertAlmostEqual(angles.get_score(A.left_knee), 1.0, places=6)

    def test_no_points_gives_dummy(self) -> None:
        angles = _angles(Points2D.create_dummy())
        self.assertEqual(angles.valid_count, 0)


def _vel_frame(t: float, **joints: float) -> Frame:
    return frame(t=t, features={Angles: scalar(Angles, {A[k]: v for k, v in joints.items()})})


class AngleVelExtractorTest(unittest.TestCase):
    def test_first_frame_is_dummy(self) -> None:
        out = AngleVelExtractor().process(_vel_frame(0.0, head=0.1))
        self.assertIn(AngleVelocity, out)
        self.assertEqual(out[AngleVelocity].valid_count, 0)

    def test_velocity_is_delta_times_frequency(self) -> None:
        ex = AngleVelExtractor()
        ex.process(_vel_frame(0.0, left_knee=0.10))
        out = ex.process(_vel_frame(1 / FPS, left_knee=0.13))
        self.assertAlmostEqual(out[AngleVelocity][A.left_knee], 0.03 * 30.0, places=4)

    def test_uses_configured_frequency_not_timestamps(self) -> None:
        settings = AngleVelExtractorSettings()
        settings.frequency = 60.0
        ex = AngleVelExtractor(settings)
        ex.process(_vel_frame(0.0, head=0.0))
        out = ex.process(_vel_frame(5.0, head=0.01))
        self.assertAlmostEqual(out[AngleVelocity][A.head], 0.6, places=4)

    def test_velocity_wraps_across_pi(self) -> None:
        ex = AngleVelExtractor()
        ex.process(_vel_frame(0.0, head=math.pi - 0.05))
        out = ex.process(_vel_frame(1 / FPS, head=-math.pi + 0.05))
        self.assertAlmostEqual(out[AngleVelocity][A.head], 0.1 * 30.0, places=3)

    def test_reappearing_joint_has_no_velocity(self) -> None:
        ex = AngleVelExtractor()
        ex.process(_vel_frame(0.0, head=0.1))
        ex.process(_vel_frame(1 / FPS, left_knee=0.2))         # head occluded
        out = ex.process(_vel_frame(2 / FPS, head=0.4, left_knee=0.2))
        self.assertTrue(math.isnan(out[AngleVelocity][A.head]))
        self.assertEqual(out[AngleVelocity].get_score(A.head), 0.0)
        self.assertAlmostEqual(out[AngleVelocity][A.left_knee], 0.0, places=6)

    def test_reset_behaves_like_a_fresh_extractor(self) -> None:
        ex = AngleVelExtractor()
        ex.process(_vel_frame(0.0, head=0.1))
        ex.reset()
        out = ex.process(_vel_frame(1 / FPS, head=0.9))
        self.assertEqual(out[AngleVelocity].valid_count, 0)


if __name__ == "__main__":
    unittest.main()
