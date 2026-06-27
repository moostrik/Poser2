"""Tests for the compositor's motor-rpm crossfade weights."""

import math
import unittest

import numpy as np

from apps.white_space.light.render import crossfade_weights, CROSSFADE_DEADZONE

IR, LR, XR = 7.0, 34.0, 600.0   # idle / low / high_cross rpm (studio defaults)


class CrossfadeWeightsTest(unittest.TestCase):
    def test_idle_only_below_low_band(self) -> None:
        self.assertEqual(crossfade_weights(0.0, IR, LR, XR), (1.0, 0.0, 0.0))
        wi, wl, wh = crossfade_weights(IR, IR, LR, XR)   # low has not started at idle_rpm
        self.assertAlmostEqual(wi, 1.0); self.assertAlmostEqual(wl, 0.0); self.assertAlmostEqual(wh, 0.0)

    def test_low_full_at_low_rpm(self) -> None:
        wi, wl, wh = crossfade_weights(LR, IR, LR, XR)   # fully on a bit before low_rpm (deadzone)
        self.assertAlmostEqual(wl, 1.0); self.assertAlmostEqual(wi, 0.0); self.assertAlmostEqual(wh, 0.0)

    def test_high_full_at_and_above_cross_rpm(self) -> None:
        self.assertAlmostEqual(crossfade_weights(XR, IR, LR, XR)[2], 1.0)
        self.assertAlmostEqual(crossfade_weights(XR * 2, IR, LR, XR)[2], 1.0)

    def test_partition_sums_to_one(self) -> None:
        for rpm in np.linspace(0.0, XR * 1.2, 200):
            wi, wl, wh = crossfade_weights(float(rpm), IR, LR, XR)
            self.assertAlmostEqual(wi + wl + wh, 1.0, places=6)
            for w in (wi, wl, wh):
                self.assertGreaterEqual(w, 0.0)
                self.assertLessEqual(w, 1.0)

    def test_dead_zone_no_neighbour_bleed(self) -> None:
        # within +deadzone of a mode rpm, the next comp stays fully off (no bleed under rpm jitter)
        jit = 1.0 + CROSSFADE_DEADZONE * 0.5
        for f in (1.0, jit):                        # at/just above low_rpm → high must be off
            wi, wl, wh = crossfade_weights(LR * f, IR, LR, XR)
            self.assertEqual(wh, 0.0)
            self.assertAlmostEqual(wl, 1.0)
        for f in (1.0, jit):                        # at/just above idle_rpm → low must be off
            wi, wl, wh = crossfade_weights(IR * f, IR, LR, XR)
            self.assertEqual(wl, 0.0)
            self.assertAlmostEqual(wi, 1.0)

    def test_bands_blend_two_slots(self) -> None:
        mid_low = (IR + LR * (1.0 - CROSSFADE_DEADZONE)) / 2.0   # idle↔low mix, no high
        wi, wl, wh = crossfade_weights(mid_low, IR, LR, XR)
        self.assertEqual(wh, 0.0)
        self.assertGreater(wi, 0.0); self.assertGreater(wl, 0.0)
        mid_high = (LR + XR) / 2.0                              # low↔high mix
        wi, wl, wh = crossfade_weights(mid_high, IR, LR, XR)
        self.assertGreater(wl, 0.0); self.assertGreater(wh, 0.0)


class LightOffsetShiftTest(unittest.TestCase):
    """Documents the light_offset → pixel-shift convention used in Render._compose."""

    @staticmethod
    def _shift(offset: float, resolution: int) -> int:
        return int(round(offset / (2.0 * np.pi) * resolution)) % resolution

    def test_zero_offset_is_identity(self) -> None:
        self.assertEqual(self._shift(0.0, 3600), 0)

    def test_pi_is_half_ring(self) -> None:
        self.assertEqual(self._shift(math.pi, 3600), 1800)


if __name__ == "__main__":
    unittest.main()
