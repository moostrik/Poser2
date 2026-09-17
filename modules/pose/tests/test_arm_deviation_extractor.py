"""Tests for ArmDeviationExtractor — the position of the arm joint furthest from neutral in a degree range, top-N aggregated."""

import math
import unittest

import numpy as np

from modules.pose.frame import Frame
from modules.pose.features import Angles, AngleLandmark, ArmDeviation
from modules.pose.nodes import ArmDeviationExtractor, ArmDeviationExtractorSettings

MIN, MAX = 18.0, 54.0        # the range (°)


def _frame(**arms: float) -> Frame:
    """A pose frame with the given arm angles (degrees, calibrated: 0 hanging / straight); every other angle NaN."""
    n = len(AngleLandmark)
    values = np.full(n, np.nan, dtype=np.float32)
    scores = np.zeros(n, dtype=np.float32)
    for name, degrees in arms.items():
        values[AngleLandmark[name]] = math.radians(degrees)
        scores[AngleLandmark[name]] = 1.0
    return Frame(track_id=0, cam_id=0, features={Angles: Angles(values, scores)})


def _settings(n_top: int = 1) -> ArmDeviationExtractorSettings:
    cfg = ArmDeviationExtractorSettings()
    cfg.min_degrees = MIN
    cfg.max_degrees = MAX
    cfg.n_top = n_top
    return cfg


class ArmDeviationExtractorTest(unittest.TestCase):
    def _arms(self, frame: Frame, n_top: int = 1) -> float:
        return ArmDeviationExtractor(_settings(n_top)).process(frame)[ArmDeviation].value

    def test_hanging_is_zero(self) -> None:
        f = _frame(left_shoulder=0.0, right_shoulder=0.0, left_elbow=0.0, right_elbow=0.0)
        self.assertAlmostEqual(self._arms(f), 0.0, places=5)

    def test_within_min_is_still_zero(self) -> None:
        self.assertAlmostEqual(self._arms(_frame(left_shoulder=MIN)), 0.0, places=5)

    def test_half_way_up_the_range_is_half(self) -> None:
        self.assertAlmostEqual(self._arms(_frame(left_shoulder=(MIN + MAX) / 2.0)), 0.5, places=5)

    def test_from_max_on_is_one(self) -> None:
        self.assertAlmostEqual(self._arms(_frame(left_shoulder=MAX)), 1.0, places=5)
        self.assertAlmostEqual(self._arms(_frame(left_shoulder=90.0)), 1.0, places=5)

    def test_top_two_averages_the_movement_before_the_range(self) -> None:
        f = _frame(left_shoulder=MAX, right_shoulder=0.0, left_elbow=0.0, right_elbow=0.0)
        expected = (MAX / 2.0 - MIN) / (MAX - MIN)          # a 27° mean movement
        self.assertAlmostEqual(self._arms(f, n_top=2), expected, places=5)

    def test_a_folded_elbow_alone_registers(self) -> None:
        f = _frame(left_shoulder=0.0, right_shoulder=0.0, left_elbow=MAX, right_elbow=0.0)
        self.assertAlmostEqual(self._arms(f), 1.0, places=5)

    def test_sign_is_ignored(self) -> None:
        self.assertAlmostEqual(self._arms(_frame(right_shoulder=-MAX)), 1.0, places=5)

    def test_nan_joints_are_ignored(self) -> None:
        f = _frame(right_elbow=MAX)                          # three joints NaN
        self.assertAlmostEqual(self._arms(f, n_top=2), 1.0, places=5)

    def test_all_nan_leaves_feature_absent(self) -> None:
        out = ArmDeviationExtractor(_settings()).process(_frame())
        self.assertNotIn(ArmDeviation, out)
        self.assertTrue(math.isnan(out[ArmDeviation].value))
        self.assertEqual(out[ArmDeviation].score, 0.0)

    def test_score_is_mean_of_used_joints(self) -> None:
        out = ArmDeviationExtractor(_settings()).process(_frame(left_shoulder=5.0, right_shoulder=5.0))
        self.assertAlmostEqual(out[ArmDeviation].score, 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
