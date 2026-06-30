"""Tests for the compositor's time-driven mode→slot crossfade."""

import unittest

import numpy as np

from apps.white_space.light.layers.crossfade import _slot_for_mode, transition_weights
from apps.white_space.light.motor import MotorMode

IDLE, LOW, HIGH = 0, 1, 2   # slot indices


class SlotForModeTest(unittest.TestCase):
    def test_mode_to_slot_mapping(self) -> None:
        self.assertEqual(_slot_for_mode(MotorMode.STOPPED), IDLE)
        self.assertEqual(_slot_for_mode(MotorMode.IDLE), IDLE)   # stopped & idle share the idle slot
        self.assertEqual(_slot_for_mode(MotorMode.LOW), LOW)
        self.assertEqual(_slot_for_mode(MotorMode.HIGH), HIGH)


class TransitionWeightsTest(unittest.TestCase):
    FROM = (1.0, 0.0, 0.0)   # resting on idle

    def test_start_returns_from_weights(self) -> None:
        self.assertEqual(transition_weights(self.FROM, LOW, 0.0), self.FROM)
        self.assertEqual(transition_weights(self.FROM, LOW, -0.5), self.FROM)   # clamped

    def test_end_is_one_hot_target(self) -> None:
        self.assertEqual(transition_weights(self.FROM, LOW, 1.0), (0.0, 1.0, 0.0))
        self.assertEqual(transition_weights(self.FROM, HIGH, 2.0), (0.0, 0.0, 1.0))   # clamped

    def test_partition_sums_to_one(self) -> None:
        for t in np.linspace(0.0, 1.0, 50):
            for target in (IDLE, LOW, HIGH):
                w = transition_weights(self.FROM, target, float(t))
                self.assertAlmostEqual(sum(w), 1.0, places=6)
                for x in w:
                    self.assertGreaterEqual(x, 0.0)
                    self.assertLessEqual(x, 1.0)

    def test_midpoint_is_eased_half(self) -> None:
        # sine ease at t=0.5 → 0.5, so an idle→low transition sits half-and-half
        wi, wl, wh = transition_weights(self.FROM, LOW, 0.5)
        self.assertAlmostEqual(wi, 0.5); self.assertAlmostEqual(wl, 0.5); self.assertAlmostEqual(wh, 0.0)

    def test_direct_idle_to_high_skips_low(self) -> None:
        wi, wl, wh = transition_weights(self.FROM, HIGH, 0.5)
        self.assertAlmostEqual(wl, 0.0)                       # low never lights up
        self.assertAlmostEqual(wi, 0.5); self.assertAlmostEqual(wh, 0.5)

    def test_up_down_progress_from_duration(self) -> None:
        # The layer computes t = elapsed / duration; up vs down picks a different duration,
        # so the same elapsed time yields different progress (and thus different weights).
        up, down = 2.0, 0.5
        elapsed = 0.5
        slow = transition_weights(self.FROM, LOW, elapsed / up)     # t = 0.25
        fast = transition_weights(self.FROM, LOW, elapsed / down)   # t = 1.0 → settled
        self.assertLess(slow[1], fast[1])
        self.assertEqual(fast, (0.0, 1.0, 0.0))


class LightPhaseShiftTest(unittest.TestCase):
    """Documents the light_phase → pixel-shift convention used in Crossfade._draw."""

    @staticmethod
    def _shift(phase: float, resolution: int) -> int:
        return int(round(phase * resolution)) % resolution

    def test_zero_phase_is_identity(self) -> None:
        self.assertEqual(self._shift(0.0, 3600), 0)

    def test_half_turn_is_half_ring(self) -> None:
        self.assertEqual(self._shift(0.5, 3600), 1800)


if __name__ == "__main__":
    unittest.main()
