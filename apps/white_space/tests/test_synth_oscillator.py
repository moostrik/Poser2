"""Tests for the light synth's oscillator: the pulse as lines of the set width, the travel, the
interval opening from position 0, and the no-jumps rule (docs/LIGHT_SYNTH.md)."""

import math
import unittest

import numpy as np

from apps.white_space.light.synth import Oscillator

STEP = 0.01                                             # degrees per position
X = np.arange(0.0, 100.0, STEP)


def runs(levels: np.ndarray) -> list[tuple[float, float]]:
    """(centre, width) in degrees of every lit run of a hard output."""
    lit = levels > 0.5
    edges = np.flatnonzero(np.diff(np.concatenate(([False], lit, [False])).astype(int)))
    return [((s + e - 1) / 2.0 * STEP, (e - s) * STEP) for s, e in zip(edges[::2], edges[1::2])]


def lines(oscillator: Oscillator, interval: float = 10.0, pulse_width: float = 0.5, phase: float = 0.0) -> np.ndarray:
    return Oscillator.pulse(oscillator.cycle(X, interval, phase), pulse_width, 1.0)


class PulseTest(unittest.TestCase):
    def test_a_line_is_the_pulse_width_of_the_interval_wide(self) -> None:
        for pulse_width in (0.1, 0.25, 0.5, 0.9):
            centre, width = runs(lines(Oscillator(), 10.0, pulse_width))[1]
            self.assertAlmostEqual(centre, 10.0, delta=2 * STEP)
            self.assertAlmostEqual(width, pulse_width * 10.0, delta=2 * STEP)

    def test_width_zero_is_dark_and_width_one_is_solid(self) -> None:
        self.assertFalse(lines(Oscillator(), pulse_width=0.0).any())
        self.assertTrue((lines(Oscillator(), pulse_width=1.0) == 1.0).all())

    def test_a_thin_line_is_drawn_not_dropped(self) -> None:
        centre, width = runs(lines(Oscillator(), 10.0, 0.005))[1]
        self.assertAlmostEqual(width, 0.05, delta=2 * STEP)

    def test_phase_places_the_lines(self) -> None:
        self.assertEqual(lines(Oscillator(), phase=0.0)[0], 1.0)          # a line centred on position 0
        gap = lines(Oscillator(), phase=0.5)
        self.assertEqual(gap[0], 0.0)                                     # a gap there
        self.assertAlmostEqual(runs(gap)[0][0], 5.0, delta=2 * STEP)
        np.testing.assert_array_equal(lines(Oscillator(), phase=1.0), lines(Oscillator(), phase=0.0))

    def test_a_hard_pulse_is_off_or_full(self) -> None:
        self.assertEqual(set(np.unique(lines(Oscillator(), 7.3, 0.37, 0.21))), {0.0, 1.0})

    def test_hardness_shapes_the_flanks_only(self) -> None:
        cycle = Oscillator().cycle(X, 10.0, 0.0)
        soft = Oscillator.pulse(cycle, 0.4, 0.3)
        self.assertEqual(soft[0], 1.0)                                    # the centre of a line stays full
        self.assertEqual(soft[int(5.0 / STEP)], 0.0)                      # the centre of a gap stays off
        self.assertTrue(((soft > 0.0) & (soft < 1.0)).any())
        self.assertFalse(Oscillator.pulse(cycle, 0.0, 0.0).any())         # width 0 stays dark
        self.assertTrue((Oscillator.pulse(cycle, 1.0, 0.0) == 1.0).all())  # width 1 stays solid

    def test_softest_at_half_width_is_a_sine(self) -> None:
        cycle = Oscillator().cycle(X, 10.0, 0.0)
        np.testing.assert_allclose(Oscillator.pulse(cycle, 0.5, 0.0), 0.5 + 0.5 * np.cos(math.tau * cycle), atol=1e-6)

    def test_a_pulse_width_per_position(self) -> None:
        widths = np.where(X < 50.0, 0.5, 0.2)
        found = runs(Oscillator.pulse(Oscillator().cycle(X, 10.0, 0.0), widths, 1.0))
        self.assertAlmostEqual(found[2][1], 5.0, delta=2 * STEP)
        self.assertAlmostEqual(found[7][1], 2.0, delta=2 * STEP)


class SineTest(unittest.TestCase):
    def test_the_sine_swings_both_ways_by_its_level(self) -> None:
        wave = Oscillator.sine(Oscillator().cycle(X, 10.0, 0.0), 0.5)
        self.assertAlmostEqual(float(wave[0]), 0.5, places=6)             # highest at the whole cycle
        self.assertAlmostEqual(float(wave.min()), -0.5, places=4)
        self.assertFalse(Oscillator.sine(Oscillator().cycle(X, 10.0, 0.0), 0.0).any())     # level 0 is silent


class TravelTest(unittest.TestCase):
    def test_lines_travel_at_the_speed_whatever_the_interval(self) -> None:
        for interval in (10.0, 20.0):
            oscillator = Oscillator()
            for _ in range(100):
                oscillator.update(0.01, interval, 2.0)                     # one second at 2 deg/s
            self.assertAlmostEqual(runs(lines(oscillator, interval, 0.2))[1][0], interval + 2.0, delta=3 * STEP)

    def test_a_negative_speed_travels_inward(self) -> None:
        oscillator = Oscillator()
        oscillator.update(1.0, 10.0, -2.0)
        self.assertAlmostEqual(runs(lines(oscillator, 10.0, 0.2))[0][0], 8.0, delta=3 * STEP)

    def test_a_whole_interval_travelled_is_the_same_picture(self) -> None:
        oscillator = Oscillator()
        for _ in range(500):
            oscillator.update(0.01, 10.0, 2.0)                             # five seconds: one interval
        differing = np.count_nonzero(lines(oscillator) != lines(Oscillator()))
        self.assertLessEqual(differing, 22)                                # a pixel on an edge may round either way

    def test_a_change_of_interval_opens_the_lines_from_position_zero(self) -> None:
        oscillator = Oscillator()
        for _ in range(173):
            oscillator.update(0.01, 10.0, 25.0)                            # far travelled: 4.325 intervals
        narrow = [c for c, _ in runs(lines(oscillator, 10.0, 0.2))]
        wide = [c for c, _ in runs(lines(oscillator, 20.0, 0.2))]
        for a, b in zip(narrow[:4], wide[:4]):
            self.assertAlmostEqual(b, 2.0 * a, delta=4 * STEP)             # every line twice as far, none elsewhere


class NoJumpsTest(unittest.TestCase):
    """A smooth change of an input is a smooth change of the picture: between two small steps only
    the pixels at the edges change, a pixel or two per edge, and nothing appears or goes at once."""
    EDGES = 2 * 11                                                         # at most 11 lines over X at interval ≥ 10
    LIMIT = 3 * EDGES                                                      # pixels changed per step; a jump is hundreds

    def _sweep(self, pictures: list[np.ndarray]) -> int:
        return max(int(np.count_nonzero(a != b)) for a, b in zip(pictures, pictures[1:]))

    def test_the_pulse_width_from_nothing_to_solid(self) -> None:
        pictures = [lines(Oscillator(), 10.0, w) for w in np.linspace(0.0, 1.0, 1001)]
        self.assertLessEqual(self._sweep(pictures), self.LIMIT)

    def test_the_phase_round_the_cycle(self) -> None:
        pictures = [lines(Oscillator(), 10.0, 0.3, p) for p in np.linspace(-0.5, 0.5, 1001)]
        self.assertLessEqual(self._sweep(pictures), self.LIMIT)

    def test_the_interval(self) -> None:
        pictures = [lines(Oscillator(), i, 0.3) for i in np.linspace(10.0, 12.0, 2001)]
        self.assertLessEqual(self._sweep(pictures), self.LIMIT)

    def test_the_travel(self) -> None:
        oscillator, pictures = Oscillator(), []
        for _ in range(1000):
            oscillator.update(0.001, 10.0, 8.0)
            pictures.append(lines(oscillator, 10.0, 0.3))
        self.assertLessEqual(self._sweep(pictures), self.LIMIT)


if __name__ == "__main__":
    unittest.main()
