"""Tests for AngleCalibrator — the angles mapped from two reference poses: neutral to 0, raised to π."""

import math
import unittest

import numpy as np

from modules.pose.features import Angles, AngleLandmark
from modules.pose.frame import Frame
from modules.pose.nodes import AngleCalibrator, AngleCalibratorSettings

A = AngleLandmark


def _frame(**joints: float) -> Frame:
    n = len(AngleLandmark)
    values = np.full(n, np.nan, dtype=np.float32)
    scores = np.zeros(n, dtype=np.float32)
    for name, angle in joints.items():
        values[A[name]] = angle
        scores[A[name]] = 0.7
    return Frame(track_id=0, cam_id=0, features={Angles: Angles(values, scores)})


def _settings(**values: float) -> AngleCalibratorSettings:
    cfg = AngleCalibratorSettings()
    cfg.shoulder_neutral, cfg.shoulder_raised = -0.2, 2.5      # a travel of 2.7 to straight up
    cfg.elbow_neutral, cfg.elbow_raised = -2.8, 2.9            # relaxed straight: bent one way hanging, the other raised
    cfg.hip_neutral, cfg.knee_neutral = 3.0, -3.1
    for name, value in values.items():
        setattr(cfg, name, value)
    return cfg


class AngleCalibratorTest(unittest.TestCase):
    def _out(self, frame: Frame, **values: float) -> Angles:
        return AngleCalibrator(_settings(**values)).process(frame)[Angles]

    def test_neutral_reads_zero_on_every_joint(self) -> None:
        out = self._out(_frame(left_shoulder=-0.2, right_shoulder=-0.2, left_elbow=-2.8, right_elbow=-2.8,
                               left_hip=3.0, right_hip=3.0, left_knee=-3.1, right_knee=-3.1))
        for joint in (A.left_shoulder, A.right_shoulder, A.left_elbow, A.right_elbow, A.left_hip, A.right_hip, A.left_knee, A.right_knee):
            with self.subTest(joint=joint.name):
                self.assertAlmostEqual(out[joint], 0.0, places=5)

    def test_the_raised_shoulder_reads_pi(self) -> None:
        out = self._out(_frame(left_shoulder=2.5, right_shoulder=2.5))
        self.assertAlmostEqual(abs(out[A.left_shoulder]), math.pi, places=5)
        self.assertAlmostEqual(abs(out[A.right_shoulder]), math.pi, places=5)

    def test_the_raised_pose_reads_zero_at_the_elbow(self) -> None:
        # Arms raised: the shoulder at its raised reading, the elbow straight but relaxed the other way.
        out = self._out(_frame(left_shoulder=2.5, left_elbow=2.9))
        self.assertAlmostEqual(out[A.left_elbow], 0.0, places=5)

    def test_the_elbows_zero_slides_with_the_lift(self) -> None:
        # Half way up the elbow's zero is half way between its two readings (a slide of 0.583 the short way round).
        slide = math.atan2(math.sin(2.9 + 2.8), math.cos(2.9 + 2.8))
        out = self._out(_frame(left_shoulder=-0.2 - 1.35, left_elbow=-2.8 + slide / 2))
        self.assertAlmostEqual(out[A.left_elbow], 0.0, places=5)

    def test_a_folded_elbow_reads_its_geometric_fold(self) -> None:
        out = self._out(_frame(left_shoulder=-0.2, left_elbow=-2.8 + 1.0))       # bent 1.0 from straight, hanging
        self.assertAlmostEqual(out[A.left_elbow], 1.0, places=5)
        out = self._out(_frame(left_shoulder=2.5, left_elbow=2.9 - 1.0))         # bent 1.0 from straight, raised
        self.assertAlmostEqual(out[A.left_elbow], -1.0, places=5)

    def test_a_missing_shoulder_leaves_the_elbow_at_its_neutral_zero(self) -> None:
        out = self._out(_frame(left_elbow=-2.8))
        self.assertAlmostEqual(out[A.left_elbow], 0.0, places=5)

    def test_raised_reads_its_travel_with_raised_off(self) -> None:
        out = self._out(_frame(left_shoulder=2.5), raised=False)
        self.assertAlmostEqual(out[A.left_shoulder], 2.7, places=5)

    def test_neutral_off_leaves_the_reading(self) -> None:
        out = self._out(_frame(left_shoulder=0.3, left_hip=1.0), neutral=False, raised=False)
        self.assertAlmostEqual(out[A.left_shoulder], 0.3, places=5)
        self.assertAlmostEqual(out[A.left_hip], 1.0, places=5)

    def test_half_way_reads_half_pi_and_keeps_its_sign(self) -> None:
        out = self._out(_frame(left_shoulder=-0.2 - 1.35, right_shoulder=-0.2 + 1.35))   # the travel either way
        self.assertAlmostEqual(out[A.left_shoulder], -math.pi / 2, places=5)
        self.assertAlmostEqual(out[A.right_shoulder], math.pi / 2, places=5)

    def test_the_legs_are_neutralised_only(self) -> None:
        out = self._out(_frame(left_hip=3.0 - 0.5, left_knee=-3.1 + 0.4))
        self.assertAlmostEqual(out[A.left_hip], -0.5, places=5)
        self.assertAlmostEqual(out[A.left_knee], 0.4, places=5)

    def test_the_difference_wraps(self) -> None:
        out = self._out(_frame(left_knee=3.1))                     # 0.08 past −π from the neutral at −3.1
        self.assertAlmostEqual(out[A.left_knee], 3.1 - (-3.1) - math.tau, places=5)

    def test_a_changed_neutral_moves_that_joint_alone(self) -> None:
        # Except the elbow, whose zero follows the shoulder's lift by design.
        before = self._out(_frame(left_shoulder=0.3, left_elbow=-2.0, left_hip=2.0))
        after = self._out(_frame(left_shoulder=0.3, left_elbow=-2.0, left_hip=2.0), hip_neutral=2.5)
        self.assertNotAlmostEqual(after[A.left_hip], before[A.left_hip], places=3)
        self.assertAlmostEqual(after[A.left_shoulder], before[A.left_shoulder], places=5)
        self.assertAlmostEqual(after[A.left_elbow], before[A.left_elbow], places=5)

    def test_the_head_is_untouched(self) -> None:
        self.assertAlmostEqual(self._out(_frame(head=0.3))[A.head], 0.3, places=5)

    def test_nan_stays_nan_with_its_score_and_scores_are_kept(self) -> None:
        out = self._out(_frame(left_shoulder=0.3))
        self.assertTrue(math.isnan(out[A.right_shoulder]))
        self.assertEqual(out.get_score(A.right_shoulder), 0.0)
        self.assertAlmostEqual(out.get_score(A.left_shoulder), 0.7, places=5)

    def test_no_angles_leaves_the_frame(self) -> None:
        frame = Frame(track_id=0, cam_id=0)
        self.assertIs(AngleCalibrator(_settings()).process(frame), frame)

    def test_the_defaults_change_nothing_but_the_neutral(self) -> None:
        # The default raised readings sit π from the shoulder's neutral (scale 1) and equal the elbow's
        # neutral (no slide): the old offsets, as settings.
        out = AngleCalibrator().process(_frame(left_shoulder=-0.15 * math.pi + 1.0, left_elbow=-0.9 * math.pi + 0.5))[Angles]
        self.assertAlmostEqual(out[A.left_shoulder], 1.0, places=5)
        self.assertAlmostEqual(out[A.left_elbow], 0.5, places=5)


if __name__ == "__main__":
    unittest.main()
