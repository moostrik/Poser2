"""Tests for ArmDeviationExtractor — joint-weighted shoulder/elbow deviation from hanging, top-N aggregated."""

import math
import unittest

import numpy as np

from modules.pose.frame import Frame
from modules.pose.features import Angles, AngleLandmark, ArmDeviation
from modules.pose.nodes import ArmDeviationExtractor, ArmDeviationExtractorSettings


def _frame(**arms: float) -> Frame:
    """A pose frame with the given arm angles (radians, calibrated: 0 hanging / straight); every other angle NaN."""
    n = len(AngleLandmark)
    values = np.full(n, np.nan, dtype=np.float32)
    scores = np.zeros(n, dtype=np.float32)
    for name, angle in arms.items():
        values[AngleLandmark[name]] = angle
        scores[AngleLandmark[name]] = 1.0
    return Frame(track_id=0, cam_id=0, features={Angles: Angles(values, scores)})


def _settings(n_top: int = 1) -> ArmDeviationExtractorSettings:
    cfg = ArmDeviationExtractorSettings()
    cfg.shoulder_degrees = 90.0
    cfg.elbow_degrees = 90.0
    cfg.n_top = n_top
    return cfg


class ArmDeviationExtractorTest(unittest.TestCase):
    def _arms(self, frame: Frame, n_top: int = 1) -> float:
        return ArmDeviationExtractor(_settings(n_top)).process(frame)[ArmDeviation].value

    def test_hanging_is_zero(self) -> None:
        f = _frame(left_shoulder=0.0, right_shoulder=0.0, left_elbow=0.0, right_elbow=0.0)
        self.assertAlmostEqual(self._arms(f), 0.0, places=5)

    def test_one_shoulder_level_registers_fully_with_top_one(self) -> None:
        f = _frame(left_shoulder=math.pi / 2.0, right_shoulder=0.0, left_elbow=0.0, right_elbow=0.0)
        self.assertAlmostEqual(self._arms(f, n_top=1), 1.0, places=5)

    def test_one_shoulder_level_is_half_with_top_two(self) -> None:
        f = _frame(left_shoulder=math.pi / 2.0, right_shoulder=0.0, left_elbow=0.0, right_elbow=0.0)
        self.assertAlmostEqual(self._arms(f, n_top=2), 0.5, places=5)

    def test_a_folded_elbow_alone_registers(self) -> None:
        f = _frame(left_shoulder=0.0, right_shoulder=0.0, left_elbow=math.pi / 2.0, right_elbow=0.0)
        self.assertAlmostEqual(self._arms(f, n_top=1), 1.0, places=5)

    def test_sign_is_ignored(self) -> None:
        f = _frame(right_shoulder=-math.pi / 2.0)
        self.assertAlmostEqual(self._arms(f), 1.0, places=5)

    def test_beyond_full_clamps(self) -> None:
        f = _frame(left_shoulder=math.pi, right_shoulder=math.pi)
        self.assertAlmostEqual(self._arms(f, n_top=2), 1.0, places=5)

    def test_nan_joints_are_ignored(self) -> None:
        f = _frame(right_elbow=math.pi / 4.0)               # three joints NaN
        self.assertAlmostEqual(self._arms(f, n_top=2), 0.5, places=5)

    def test_all_nan_leaves_feature_absent(self) -> None:
        out = ArmDeviationExtractor(_settings()).process(_frame())
        self.assertNotIn(ArmDeviation, out)
        self.assertTrue(math.isnan(out[ArmDeviation].value))
        self.assertEqual(out[ArmDeviation].score, 0.0)

    def test_score_is_mean_of_used_joints(self) -> None:
        out = ArmDeviationExtractor(_settings()).process(_frame(left_shoulder=0.1, right_shoulder=0.1))
        self.assertAlmostEqual(out[ArmDeviation].score, 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
