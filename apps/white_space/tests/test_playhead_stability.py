"""Tests for PlayheadStabilityExtractor — playhead-hit sampling, the stability formula, the
zero-crossing/​wrap guard, and azimuth/track resets."""

import math
import unittest

import numpy as np

from modules.pose.frame import Frame
from modules.pose.features import Angles, Azimuth
from apps.white_space.pose import PlayheadPhase, PlayheadStability, PlayheadStabilityExtractor, PlayheadStabilityExtractorSettings

_NUM_JOINTS = len(Angles.enum())


def _angles(value: float, score: float = 1.0) -> Angles:
    return Angles(np.full(_NUM_JOINTS, value, dtype=np.float32),
                  np.full(_NUM_JOINTS, score, dtype=np.float32))


def _frame(phase: float, azimuth: float, angle_value: float) -> Frame:
    features = {Angles: _angles(angle_value)}
    if not math.isnan(azimuth):
        features[Azimuth] = Azimuth.from_value(azimuth)
    if not math.isnan(phase):
        features[PlayheadPhase] = PlayheadPhase.from_value(phase)
    return Frame(track_id=0, cam_id=0, features=features)


class StabilityFormulaTest(unittest.TestCase):
    def setUp(self) -> None:
        self.extractor = PlayheadStabilityExtractor(PlayheadStabilityExtractorSettings())

    def _value(self, frame: Frame) -> float:
        return frame[PlayheadStability].value

    def test_single_sample_is_zero(self) -> None:
        frame = self._last(_drive=1, angle=0.0)
        self.assertEqual(self._value(frame), 0.0)
        self.assertEqual(frame[PlayheadStability].score, 1.0)   # present (0.0), not absent

    def test_full_buffer_identical_is_one(self) -> None:
        frame = self._last(_drive=4, angle=0.5)
        self.assertAlmostEqual(self._value(frame), 1.0, places=5)

    def test_two_identical_is_one_third(self) -> None:
        frame = self._last(_drive=2, angle=0.5)
        self.assertAlmostEqual(self._value(frame), 1.0 / 3.0, places=4)

    def test_dissimilar_current_is_near_zero(self) -> None:
        for _ in range(3):
            self._hit(angle=0.0)
        frame = self._hit(angle=math.pi)   # current maximally different from the older three
        self.assertLess(self._value(frame), 0.05)

    # --- helpers ---------------------------------------------------------
    def _hit(self, angle: float, azimuth: float = 0.0) -> Frame:
        last = None
        for phase in (2.0, 0.1, -0.1):
            last = self.extractor.process(_frame(phase, azimuth, angle))
        return last

    def _last(self, _drive: int, angle: float) -> Frame:
        frame = None
        for _ in range(_drive):
            frame = self._hit(angle)
        return frame


class CrossingGuardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.extractor = PlayheadStabilityExtractor(PlayheadStabilityExtractorSettings())

    def _hit(self, angle: float = 0.5, azimuth: float = 0.0) -> Frame:
        last = None
        for phase in (2.0, 0.1, -0.1):
            last = self.extractor.process(_frame(phase, azimuth, angle))
        return last

    def test_pi_wrap_does_not_sample(self) -> None:
        # One real hit (1 sample), then a ±π up-crossing that must be ignored, then a
        # second real hit. If the wrap had sampled, we'd see 3 samples (≈0.667).
        self._hit()
        for phase in (-3.0, 3.0):   # sweep across the −π/+π seam (|phase| > π/2 → guarded)
            self.extractor.process(_frame(phase, 0.0, 0.5))
        frame = self._hit()
        self.assertAlmostEqual(frame[PlayheadStability].value, 1.0 / 3.0, places=4)

    def test_value_holds_between_hits(self) -> None:
        self._hit()
        self._hit()                                            # 2 samples → 1/3
        held = self.extractor.process(_frame(-0.5, 0.0, 0.5))  # stays below zero — no crossing
        self.assertAlmostEqual(held[PlayheadStability].value, 1.0 / 3.0, places=4)


class ResetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.extractor = PlayheadStabilityExtractor(PlayheadStabilityExtractorSettings())   # azimuth_reset = 0.35 rad

    def _hit(self, angle: float = 0.5, azimuth: float = 0.0) -> Frame:
        last = None
        for phase in (2.0, 0.1, -0.1):
            last = self.extractor.process(_frame(phase, azimuth, angle))
        return last

    def test_azimuth_drift_clears_history(self) -> None:
        for _ in range(4):
            self._hit(azimuth=0.0)                              # build to 1.0 anchored at az 0
        moved = self.extractor.process(_frame(2.0, 1.0, 0.5))  # drift 1.0 rad > 0.35 → reset
        self.assertTrue(math.isnan(moved[PlayheadStability].value))
        self.assertEqual(moved[PlayheadStability].score, 0.0)

    def test_small_azimuth_drift_keeps_history(self) -> None:
        self._hit(azimuth=0.0)
        frame = self._hit(azimuth=0.2)                         # 0.2 rad < 0.35 → kept
        self.assertAlmostEqual(frame[PlayheadStability].value, 1.0 / 3.0, places=4)

    def test_reset_makes_value_absent(self) -> None:
        for _ in range(3):
            self._hit()
        self.extractor.reset()
        frame = self.extractor.process(_frame(2.0, 0.0, 0.5))  # no hit
        self.assertTrue(math.isnan(frame[PlayheadStability].value))
        self.assertEqual(frame[PlayheadStability].score, 0.0)


if __name__ == "__main__":
    unittest.main()
