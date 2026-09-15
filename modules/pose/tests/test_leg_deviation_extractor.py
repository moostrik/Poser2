"""Tests for LegDeviationExtractor — joint-weighted hip/knee deviation, top-N aggregated."""

import math
import unittest

import numpy as np

from modules.pose.frame import Frame
from modules.pose.features import Angles, AngleLandmark, LegDeviation
from modules.pose.nodes import LegDeviationExtractor, LegDeviationExtractorSettings


def _frame(**legs: float) -> Frame:
    """A pose frame with the given leg angles (radians); every other angle NaN."""
    n = len(AngleLandmark)
    values = np.full(n, np.nan, dtype=np.float32)
    scores = np.zeros(n, dtype=np.float32)
    for name, angle in legs.items():
        values[AngleLandmark[name]] = angle
        scores[AngleLandmark[name]] = 1.0
    return Frame(track_id=0, cam_id=0, features={Angles: Angles(values, scores)})


def _settings(n_top: int = 2, min_degrees: float = 0.0) -> LegDeviationExtractorSettings:
    cfg = LegDeviationExtractorSettings()
    cfg.min_degrees = min_degrees
    cfg.hip_max_degrees = 60.0
    cfg.knee_max_degrees = 90.0
    cfg.n_top = n_top
    return cfg


class LegDeviationExtractorTest(unittest.TestCase):
    def _legs(self, frame: Frame, n_top: int = 2, min_degrees: float = 0.0) -> float:
        return LegDeviationExtractor(_settings(n_top, min_degrees)).process(frame)[LegDeviation].value

    def test_the_range_starts_at_min_degrees(self) -> None:
        # From 30° to the knee's 90° top: a 60° bend sits half way.
        f = _frame(left_knee=math.radians(60.0))
        self.assertAlmostEqual(self._legs(f, n_top=1, min_degrees=30.0), 0.5, places=5)
        self.assertAlmostEqual(self._legs(_frame(left_knee=math.radians(30.0)), n_top=1, min_degrees=30.0), 0.0, places=5)

    def test_standing_straight_is_zero(self) -> None:
        f = _frame(left_hip=0.0, right_hip=0.0, left_knee=0.0, right_knee=0.0)
        self.assertAlmostEqual(self._legs(f), 0.0, places=5)

    def test_one_knee_at_full_registers_fully_with_top_one(self) -> None:
        f = _frame(left_hip=0.0, right_hip=0.0, left_knee=math.pi / 2.0, right_knee=0.0)
        self.assertAlmostEqual(self._legs(f, n_top=1), 1.0, places=5)

    def test_one_knee_at_full_is_half_with_top_two(self) -> None:
        f = _frame(left_hip=0.0, right_hip=0.0, left_knee=math.pi / 2.0, right_knee=0.0)
        self.assertAlmostEqual(self._legs(f, n_top=2), 0.5, places=5)

    def test_hip_is_weighted_more_than_knee(self) -> None:
        # The same angle counts for more at the hip (smaller full-deviation angle).
        hip  = _frame(left_hip=math.pi / 4.0, right_hip=0.0, left_knee=0.0, right_knee=0.0)
        knee = _frame(left_hip=0.0, right_hip=0.0, left_knee=math.pi / 4.0, right_knee=0.0)
        self.assertGreater(self._legs(hip, n_top=1), self._legs(knee, n_top=1))
        self.assertAlmostEqual(self._legs(hip, n_top=1), 0.75, places=5)
        self.assertAlmostEqual(self._legs(knee, n_top=1), 0.5, places=5)

    def test_sign_is_ignored(self) -> None:
        f = _frame(left_knee=-math.pi / 2.0)
        self.assertAlmostEqual(self._legs(f, n_top=1), 1.0, places=5)

    def test_beyond_full_clamps(self) -> None:
        f = _frame(left_hip=math.pi, right_hip=math.pi)
        self.assertAlmostEqual(self._legs(f), 1.0, places=5)

    def test_nan_joints_are_ignored(self) -> None:
        f = _frame(right_knee=math.pi / 2.0)             # three joints NaN
        self.assertAlmostEqual(self._legs(f, n_top=2), 1.0, places=5)

    def test_all_nan_leaves_feature_absent(self) -> None:
        out = LegDeviationExtractor(_settings()).process(_frame())
        self.assertNotIn(LegDeviation, out)
        self.assertTrue(math.isnan(out[LegDeviation].value))
        self.assertEqual(out[LegDeviation].score, 0.0)

    def test_score_is_mean_of_used_joints(self) -> None:
        f = _frame(left_hip=0.1, right_hip=0.1)
        out = LegDeviationExtractor(_settings()).process(f)
        self.assertAlmostEqual(out[LegDeviation].score, 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
