"""Tests for PlayheadStabilityExtractor — the three per-spot values (dwell / motion / stability),
the beat-continuous dwell ramp, motion normalization, the dwell∧motion gate on stability, the
zero-crossing/wrap guard, and azimuth resets — plus MotionTimeExtractor robustness."""

import math
import unittest

import numpy as np

from modules.pose.frame import Frame
from modules.pose.features import Angles, Azimuth, AngleVelocity, MotionTime
from modules.pose.nodes import MotionTimeExtractor
from apps.white_space.pose import (
    PlayheadElement, PlayheadOffset, PlayheadStability,
    PlayheadStabilityExtractor, PlayheadStabilityExtractorSettings,
)

_NUM_JOINTS = len(Angles.enum())


def _angles(value: float, score: float = 1.0) -> Angles:
    return Angles(np.full(_NUM_JOINTS, value, dtype=np.float32),
                  np.full(_NUM_JOINTS, score, dtype=np.float32))


def _frame(phase: float, azimuth: float = 0.0, angle: float = 0.5, motion: float = 0.0) -> Frame:
    """A LERP-stage pose frame. Timestamp is irrelevant here — dwell is beat/phase-driven and
    motion is read straight from the stamped MotionTime (fed directly)."""
    features = {Angles: _angles(angle), MotionTime: MotionTime.from_value(motion)}
    if not math.isnan(azimuth):
        features[Azimuth] = Azimuth.from_value(azimuth)
    if not math.isnan(phase):
        features[PlayheadOffset] = PlayheadOffset.from_value(phase)
    return Frame(track_id=0, cam_id=0, features=features)


class _ExtractorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ex = PlayheadStabilityExtractor(PlayheadStabilityExtractorSettings())

    def _get(self, frame: Frame, element: PlayheadElement) -> float:
        return frame[PlayheadStability].get(element)

    def _score(self, frame: Frame, element: PlayheadElement) -> float:
        return frame[PlayheadStability].get_score(element)

    def _cross(self, angle: float = 0.5, azimuth: float = 0.0, motion: float = 0.0) -> Frame:
        """Drive one playhead crossing on the spot. The 2.0 lead is a guarded (>π/2) sign flip
        that avoids a spurious −→+ crossing between calls; the tiny −0.001 tail keeps the
        post-crossing sub-beat phase ~0 so dwell reads clean integer beats."""
        last = None
        for ph in (2.0, 0.1, -0.001):
            last = self.ex.process(_frame(ph, azimuth, angle, motion))
        return last

    def _drive(self, beats: int, angle: float = 0.5, motion_after: float = 10.0) -> Frame:
        """First crossing anchors motion at 0; later crossings carry ``motion_after`` (→ motion 1)."""
        last = self._cross(angle=angle, motion=0.0)
        for _ in range(beats - 1):
            last = self._cross(angle=angle, motion=motion_after)
        return last


class DwellTest(_ExtractorTest):
    def test_absent_before_first_crossing(self) -> None:
        f = self.ex.process(_frame(2.0))                       # no crossing yet → no spot
        self.assertEqual(self._get(f, PlayheadElement.Dwell), 0.0)
        self.assertEqual(self._score(f, PlayheadElement.Dwell), 0.0)

    def test_zero_at_first_beat(self) -> None:
        f = self._cross()
        self.assertAlmostEqual(self._get(f, PlayheadElement.Dwell), 0.0, delta=0.01)
        self.assertEqual(self._score(f, PlayheadElement.Dwell), 1.0)   # present

    def test_one_at_dwell_beats(self) -> None:
        last = None
        for _ in range(4):                                     # dwell_beats default = 4
            last = self._cross()
        self.assertAlmostEqual(self._get(last, PlayheadElement.Dwell), 1.0, places=4)

    def test_holds_at_one(self) -> None:
        last = None
        for _ in range(7):
            last = self._cross()
        self.assertAlmostEqual(self._get(last, PlayheadElement.Dwell), 1.0, places=4)

    def test_ramps_between_beats(self) -> None:
        self._cross()                                          # beat 1
        # A held frame midway through the next sweep (offset ≈ −π → phase ≈ 0.5).
        mid = self.ex.process(_frame(-math.pi + 0.01, 0.0, 0.5, 0.0))
        self.assertAlmostEqual(self._get(mid, PlayheadElement.Dwell), 0.5 / 3.0, delta=0.02)


class MotionTest(_ExtractorTest):
    def test_zero_at_entry(self) -> None:
        f = self._cross(motion=3.0)                            # anchor snapshot = 3.0
        self.assertAlmostEqual(self._get(f, PlayheadElement.Motion), 0.0, delta=1e-3)

    def test_rises_and_clamps(self) -> None:
        self._cross(motion=0.0)                                # anchor = 0
        half = self.ex.process(_frame(-0.05, 0.0, 0.5, 2.5))   # (2.5-0)/5 = 0.5
        self.assertAlmostEqual(self._get(half, PlayheadElement.Motion), 0.5, delta=1e-3)
        full = self.ex.process(_frame(-0.05, 0.0, 0.5, 100.0)) # clamps to 1.0
        self.assertEqual(self._get(full, PlayheadElement.Motion), 1.0)


class GateTest(_ExtractorTest):
    def test_closed_when_dwell_low(self) -> None:
        # 3 beats: motion is high but dwell < 1 → gate closed → stability 0.
        self.assertEqual(self._get(self._drive(3), PlayheadElement.Stability), 0.0)

    def test_closed_when_no_motion(self) -> None:
        # 6 beats but motion stays 0 → gate never opens → stability 0.
        last = None
        for _ in range(6):
            last = self._cross(motion=0.0)
        self.assertEqual(self._get(last, PlayheadElement.Stability), 0.0)

    def test_rises_once_gate_opens(self) -> None:
        # Gate opens at beat 4 (dwell 1, motion 1); first gated bank → stability 1/3.
        self.assertAlmostEqual(self._get(self._drive(4), PlayheadElement.Stability), 1.0 / 3.0, delta=0.02)

    def test_full_ring_after_gate(self) -> None:
        # Beats 4,5,6 all gated & identical → ring fills → stability 1.0.
        self.assertAlmostEqual(self._get(self._drive(6), PlayheadElement.Stability), 1.0, places=4)

    def test_dissimilar_live_pose_low(self) -> None:
        self._drive(6, angle=0.0)                              # ring full of 0.0 poses, gate open
        moved = self.ex.process(_frame(-0.05, 0.0, math.pi, 10.0))  # very different live pose
        self.assertLess(self._get(moved, PlayheadElement.Stability), 0.05)


class ResetTest(_ExtractorTest):
    def test_azimuth_drift_resets_all(self) -> None:
        self._drive(6)                                         # build a full, gated spot at az 0
        moved = self.ex.process(_frame(2.0, azimuth=1.0, angle=0.5, motion=10.0))  # drift 1.0 rad > 20°
        for element in PlayheadElement:
            self.assertEqual(self._get(moved, element), 0.0)
            self.assertEqual(self._score(moved, element), 0.0)

    def test_small_drift_keeps_spot(self) -> None:
        self._drive(6)
        frame = self._cross(azimuth=0.2, motion=10.0)          # 0.2 rad ≈ 11° < 20° → kept
        self.assertAlmostEqual(self._get(frame, PlayheadElement.Stability), 1.0, places=4)


class CrossingGuardTest(_ExtractorTest):
    def test_pi_wrap_is_not_a_beat(self) -> None:
        self._cross()                                          # beat 1 (dwell ≈ 0)
        for ph in (-3.0, 3.0):                                 # cross the ±π seam (guarded)
            self.ex.process(_frame(ph, 0.0, 0.5, 0.0))
        f = self._cross()                                      # beat 2 — not beat 3
        self.assertAlmostEqual(self._get(f, PlayheadElement.Dwell), 1.0 / 3.0, delta=0.02)


class OverrideTest(unittest.TestCase):
    def test_override_forces_stability_only(self) -> None:
        settings = PlayheadStabilityExtractorSettings()
        settings.override = True
        settings.override_value = 0.7
        ex = PlayheadStabilityExtractor(settings)
        f = ex.process(_frame(2.0, 0.0, 0.5, 0.0))             # even with no spot yet
        self.assertAlmostEqual(f[PlayheadStability].get(PlayheadElement.Stability), 0.7, places=5)


def _vel_frame(velocity: float, t: float) -> Frame:
    n = len(AngleVelocity.enum())
    vel = AngleVelocity(np.full(n, velocity, dtype=np.float32), np.full(n, 1.0, dtype=np.float32))
    return Frame(track_id=0, cam_id=0, time_stamp=t, features={AngleVelocity: vel})


class MotionTimeRobustnessTest(unittest.TestCase):
    def test_inf_velocity_zero_dt_never_poisons(self) -> None:
        ex = MotionTimeExtractor()
        ex.process(_vel_frame(0.0, t=1.0))                     # establishes prev_time_stamp
        out = ex.process(_vel_frame(np.inf, t=1.0))            # inf velocity on a dt==0 tick
        self.assertFalse(math.isnan(out[MotionTime].value))
        later = ex.process(_vel_frame(1.0, t=2.0))             # normal tick still accumulates finitely
        self.assertTrue(math.isfinite(later[MotionTime].value))
        self.assertGreater(later[MotionTime].value, 0.0)

    def test_inf_velocity_positive_dt_ignored(self) -> None:
        ex = MotionTimeExtractor()
        ex.process(_vel_frame(0.0, t=1.0))
        out = ex.process(_vel_frame(np.inf, t=2.0))            # dt>0 but inf is masked out
        self.assertEqual(out[MotionTime].value, 0.0)

    def test_motion_time_stamped_from_first_frame(self) -> None:
        ex = MotionTimeExtractor()
        out = ex.process(_vel_frame(1.0, t=1.0))
        self.assertIn(MotionTime, out)
        self.assertEqual(out[MotionTime].value, 0.0)


if __name__ == "__main__":
    unittest.main()
