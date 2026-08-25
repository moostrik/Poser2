"""Tests for HauntedFlash's closest-approach guarantee (``_closest_pass``).

The width window can be stepped clean over on a fast crossing; ``_closest_pass`` guarantees one flash
on the frame where the playhead is *nearest* the pose — the local minimum of |offset| — firing in real
time on that sample whether it sits just before or just after the zero-crossing.
"""

import math
import unittest

from apps.white_space.light.layers.low.haunted_flash import _closest_pass

R = math.radians


def _fires(degrees: list[float]) -> list[bool]:
    """Run a per-tick offset sequence (degrees) through the detector; prev starts NaN (no history)."""
    prev = float("nan")
    out: list[bool] = []
    for deg in degrees:
        cur = R(deg)
        out.append(_closest_pass(prev, cur))
        prev = cur
    return out


class ClosestPassTest(unittest.TestCase):
    def test_fires_on_departing_sample_when_nearest(self) -> None:
        # 13° → 6° → −1°: the sample nearest zero is −1° (just past) → single fire there.
        self.assertEqual(_fires([13, 6, -1]), [False, False, True])

    def test_fires_on_approaching_sample_when_nearest(self) -> None:
        # 8° → 1° → −6°: the nearest sample is +1° (just before) → fires there, not "after".
        self.assertEqual(_fires([8, 1, -6]), [False, True, False])

    def test_exactly_one_fire_per_sweep(self) -> None:
        # A steady sweep fires once, on the minimum-|offset| sample (2°).
        self.assertEqual(_fires([30, 23, 16, 9, 2, -5, -12]),
                         [False, False, False, False, True, False, False])

    def test_far_side_never_fires(self) -> None:
        # |offset| stays past the near-half gate (π/2) → no guarantee flash for the opposite side.
        self.assertEqual(_fires([100, 95, 92, 95, 100]), [False] * 5)

    def test_nan_never_fires(self) -> None:
        self.assertFalse(_closest_pass(float("nan"), 0.0))     # no prev yet
        self.assertFalse(_closest_pass(0.1, float("nan")))     # offset absent (motor stopped)


if __name__ == "__main__":
    unittest.main()
