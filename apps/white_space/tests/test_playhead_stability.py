"""Tests for PlayheadStabilityExtractor — playhead-hit sampling, the stability formula, the
zero-crossing/​wrap guard, and azimuth/track resets."""

import math
import unittest

import numpy as np

from modules.pose.frame import Frame
from modules.pose.features import Angles, Azimuth
from apps.white_space.pose import PlayheadOffset, PlayheadStability, PlayheadStabilityExtractor, PlayheadStabilityExtractorSettings

_NUM_JOINTS = len(Angles.enum())


def _angles(value: float, score: float = 1.0) -> Angles:
    return Angles(np.full(_NUM_JOINTS, value, dtype=np.float32),
                  np.full(_NUM_JOINTS, score, dtype=np.float32))


def _frame(phase: float, azimuth: float, angle_value: float) -> Frame:
    features = {Angles: _angles(angle_value)}
    if not math.isnan(azimuth):
        features[Azimuth] = Azimuth.from_value(azimuth)
    if not math.isnan(phase):
        features[PlayheadOffset] = PlayheadOffset.from_value(phase)
    return Frame(track_id=0, cam_id=0, features=features)


class StabilityFormulaTest(unittest.TestCase):
    def setUp(self) -> None:
        self.extractor = PlayheadStabilityExtractor(PlayheadStabilityExtractorSettings())

    def _value(self, frame: Frame) -> float:
        return frame[PlayheadStability].value

    def test_one_banked_sweep_is_one_third(self) -> None:
        frame = self._last(_drive=1, angle=0.0)
        self.assertAlmostEqual(self._value(frame), 1.0 / 3.0, places=4)
        self.assertEqual(frame[PlayheadStability].score, 1.0)   # present, not absent

    def test_full_ring_identical_is_one(self) -> None:
        frame = self._last(_drive=3, angle=0.5)   # ring (maxlen 3) full of matching sweeps
        self.assertAlmostEqual(self._value(frame), 1.0, places=5)

    def test_two_banked_sweeps_is_two_thirds(self) -> None:
        frame = self._last(_drive=2, angle=0.5)
        self.assertAlmostEqual(self._value(frame), 2.0 / 3.0, places=4)

    def test_dissimilar_live_pose_is_near_zero(self) -> None:
        for _ in range(3):
            self._hit(angle=0.0)               # bank three steady sweeps → ring full of 0.0
        # A different live pose, read mid-sweep before the next crossing banks it: the value
        # drops immediately (no rotation of lag) because it scores the live pose vs history.
        frame = self.extractor.process(_frame(-0.5, 0.0, math.pi))   # held frame, no crossing
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
        # One real hit (1 banked sweep), then a ±π up-crossing that must be ignored, then a
        # second real hit → 2 banked sweeps (2/3). If the wrap had banked one too, we'd see 1.0.
        self._hit()
        for phase in (-3.0, 3.0):   # sweep across the −π/+π seam (|phase| > π/2 → guarded)
            self.extractor.process(_frame(phase, 0.0, 0.5))
        frame = self._hit()
        self.assertAlmostEqual(frame[PlayheadStability].value, 2.0 / 3.0, places=4)

    def test_held_pose_keeps_value(self) -> None:
        self._hit()
        self._hit()                                            # 2 banked sweeps → 2/3
        held = self.extractor.process(_frame(-0.5, 0.0, 0.5))  # stays below zero — no crossing
        self.assertAlmostEqual(held[PlayheadStability].value, 2.0 / 3.0, places=4)


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
        self.assertEqual(moved[PlayheadStability].value, 0.0)
        self.assertEqual(moved[PlayheadStability].score, 0.0)

    def test_small_azimuth_drift_keeps_history(self) -> None:
        self._hit(azimuth=0.0)
        frame = self._hit(azimuth=0.2)                         # 0.2 rad < 0.35 → kept (2 banked)
        self.assertAlmostEqual(frame[PlayheadStability].value, 2.0 / 3.0, places=4)

    def test_reset_makes_value_zero(self) -> None:
        for _ in range(3):
            self._hit()
        self.extractor.reset()
        frame = self.extractor.process(_frame(2.0, 0.0, 0.5))  # no hit
        self.assertEqual(frame[PlayheadStability].value, 0.0)
        self.assertEqual(frame[PlayheadStability].score, 0.0)


if __name__ == "__main__":
    unittest.main()
