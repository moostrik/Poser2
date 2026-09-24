"""Tests for the light synth's strobe: the rate's quantization and the gate, whole lines on or off
on the clock's shared tick grid (docs/LIGHT_SYNTH.md, *The strobe*)."""

import unittest

import numpy as np

from apps.white_space.light.synth import Strobe

FPS = 32
LINES = np.arange(0, 6, dtype=np.float64)       # six standing lines, counted from the person


def dark_ticks(rate: int, width: float, ticks: int = 64, phase: float = 0.0, spread: float = 0.0,
               line: float = 0.0) -> list[int]:
    return [t for t in range(ticks)
            if Strobe.gate(t, np.array([line]), rate, width, phase, spread, FPS)[0] == 0.0]


class StrobeRateTest(unittest.TestCase):
    def test_off_below_a_half(self) -> None:
        for value in (0.0, 0.2, 0.49, -3.0):
            self.assertEqual(Strobe.rate(value, FPS), 0, value)

    def test_the_nearest_power_of_two_on_a_log_scale(self) -> None:
        for value, rate in ((0.5, 1), (1.0, 1), (1.4, 1), (1.5, 2), (2.0, 2), (3.0, 4), (4.0, 4), (5.0, 4), (6.0, 8), (12.0, 16), (16.0, 16)):
            self.assertEqual(Strobe.rate(value, FPS), rate, value)

    def test_capped_at_every_other_tick(self) -> None:
        self.assertEqual(Strobe.rate(20.0, FPS), 16)
        self.assertEqual(Strobe.rate(100.0, FPS), 16)
        self.assertEqual(Strobe.rate(16.0, 30), 8)             # 30 fps: the largest power of two within 15
        self.assertEqual(Strobe.rate(4.0, 8), 4)

    def test_the_period_is_whole_and_nested_at_a_power_of_two_rate(self) -> None:
        self.assertEqual([Strobe.period(r, FPS) for r in (1, 2, 4, 8, 16)], [32, 16, 8, 4, 2])


class StrobeGateTest(unittest.TestCase):
    def test_rate_zero_is_always_on(self) -> None:
        for t in range(40):
            np.testing.assert_array_equal(Strobe.gate(t, LINES, 0, 0.0, 0.3, 0.5, FPS), 1.0)

    def test_one_dark_tick_per_cycle_is_the_last(self) -> None:
        self.assertEqual(dark_ticks(1, 31 / 32), [31, 63])
        self.assertEqual(dark_ticks(4, 7 / 8), [7, 15, 23, 31, 39, 47, 55, 63])

    def test_sixteen_at_a_half_is_every_other_tick(self) -> None:
        self.assertEqual(dark_ticks(16, 0.5), list(range(1, 64, 2)))

    def test_the_grids_nest(self) -> None:
        for slow, fast in ((1, 2), (2, 4), (4, 8), (8, 16)):
            T_slow, T_fast = Strobe.period(slow, FPS), Strobe.period(fast, FPS)
            self.assertLessEqual(set(dark_ticks(slow, 1.0 - 1.0 / T_slow)), set(dark_ticks(fast, 1.0 - 1.0 / T_fast)))

    def test_the_width_is_the_lit_part(self) -> None:
        self.assertEqual(len(dark_ticks(4, 0.0, ticks=8)), 8)
        self.assertEqual(len(dark_ticks(4, 1.0, ticks=8)), 0)
        self.assertEqual(dark_ticks(4, 0.5, ticks=8), [4, 5, 6, 7])

    def test_the_phase_shifts_the_cycle(self) -> None:
        self.assertEqual(dark_ticks(4, 7 / 8, ticks=8, phase=0.25), [1])         # two ticks later at T 8

    def test_the_spread_delays_each_line_by_its_count(self) -> None:
        for k in range(4):
            self.assertEqual(dark_ticks(4, 7 / 8, ticks=16, spread=1 / 8, line=k), [t for t in range(16) if (t - k) % 8 == 7], k)
        self.assertEqual(dark_ticks(4, 7 / 8, ticks=16, spread=-1 / 8, line=1), [6, 14])   # inward: earlier

    def test_a_gate_is_off_or_full(self) -> None:
        for t in range(16):
            self.assertTrue(np.isin(Strobe.gate(t, LINES, 4, 0.6, 0.1, 0.37, FPS), (0.0, 1.0)).all())


if __name__ == "__main__":
    unittest.main()
