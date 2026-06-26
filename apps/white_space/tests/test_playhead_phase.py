"""Tests for PlayheadPhaseExtractor — signed [-1,1] phase of a pose's azimuth vs the playhead."""

import math
import unittest

from modules.pose.frame import Frame
from modules.pose.features import Azimuth
from apps.white_space.pose import PlayheadPhase, PlayheadPhaseExtractor


def _frame(azimuth: float) -> Frame:
    features = {} if math.isnan(azimuth) else {Azimuth: Azimuth.from_value(azimuth)}
    return Frame(track_id=0, cam_id=0, features=features)


class PlayheadPhaseExtractorTest(unittest.TestCase):
    def _phase(self, azimuth: float, playhead: float) -> float:
        node = PlayheadPhaseExtractor(lambda: playhead)
        return node.process(_frame(azimuth))[PlayheadPhase].value

    def test_on_playhead_is_zero(self) -> None:
        self.assertAlmostEqual(self._phase(0.5, 0.5), 0.0, places=5)

    def test_quarter_ahead_is_half(self) -> None:
        # azimuth a quarter-revolution ahead of the playhead (sweep dir) → +0.5
        self.assertAlmostEqual(self._phase(0.5, 0.25), 0.5, places=5)

    def test_quarter_behind_is_negative_half(self) -> None:
        # azimuth a quarter-revolution behind (playhead just passed) → -0.5
        self.assertAlmostEqual(self._phase(0.25, 0.5), -0.5, places=5)

    def test_opposite_side_is_one(self) -> None:
        # half a revolution apart → ±1 (wraps to the +1 boundary here)
        self.assertAlmostEqual(abs(self._phase(0.5, 0.0)), 1.0, places=5)

    def test_wraps_across_zero(self) -> None:
        # azimuth 0.05, playhead 0.95 → 0.10 ahead → +0.2
        self.assertAlmostEqual(self._phase(0.05, 0.95), 0.2, places=5)

    def test_missing_azimuth_leaves_phase_absent(self) -> None:
        out = PlayheadPhaseExtractor(lambda: 0.5).process(_frame(float("nan")))
        self.assertNotIn(PlayheadPhase, out)
        self.assertTrue(math.isnan(out[PlayheadPhase].value))
        self.assertEqual(out[PlayheadPhase].score, 0.0)

    def test_nan_playhead_leaves_phase_absent(self) -> None:
        out = PlayheadPhaseExtractor(lambda: float("nan")).process(_frame(0.5))
        self.assertNotIn(PlayheadPhase, out)
        self.assertTrue(math.isnan(out[PlayheadPhase].value))


if __name__ == "__main__":
    unittest.main()
