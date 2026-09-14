"""Tests for LinePattern: the two-drawbar oscillator thresholded into lines, and the visibility
morphology."""

import math
import unittest

import numpy as np

from apps.white_space.light.layers import LinePattern, Waveform


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, length) of every lit run (no wrap handling)."""
    edges = np.flatnonzero(np.diff(np.concatenate(([False], mask, [False])).astype(int)))
    return [(int(s), int(e - s)) for s, e in zip(edges[::2], edges[1::2])]


def mask(n: int, *spans: tuple[int, int]) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for start, stop in spans:
        m[start:stop] = True
    return m


class LinesTest(unittest.TestCase):
    X = np.arange(400, dtype=np.float64)
    INTERVAL = 40.0
    PHASE = 0.8875      # line centres at x = 35.5, 75.5, …: every edge falls between two pixels

    def _lines(self, fundamental: float, harmonic: float, waveform: int = Waveform.SINE,
               cutoff: int = 2, overtone_phase: float = 0.0, phase: float | None = None) -> np.ndarray:
        return LinePattern.lines(self.X, self.INTERVAL, int(waveform), fundamental, harmonic, cutoff,
                                 overtone_phase, self.PHASE if phase is None else phase)

    def _inner(self, lit: np.ndarray) -> list[tuple[int, int]]:
        return [r for r in runs(lit) if r[0] > 0 and r[0] + r[1] < lit.size]

    def test_both_drawbars_in_is_dark_and_both_out_is_solid(self) -> None:
        for waveform in Waveform:
            with self.subTest(waveform=waveform.name):
                self.assertFalse(self._lines(0.0, 0.0, waveform).any())
                self.assertTrue(self._lines(1.0, 1.0, waveform, overtone_phase=0.3).all())

    def test_the_fundamental_alone_out_lights_half_the_interval(self) -> None:
        for waveform in Waveform:
            with self.subTest(waveform=waveform.name):
                inner = self._inner(self._lines(1.0, 0.0, waveform))
                self.assertGreater(len(inner), 5)
                self.assertEqual({length for _, length in inner}, {20})

    def test_the_fundamental_grows_a_line_at_every_whole_u(self) -> None:
        inner = self._inner(self._lines(0.5, 0.0))
        self.assertGreater(len(inner), 5)
        for start, length in inner:
            self.assertAlmostEqual(length, math.acos(0.5) / math.pi * self.INTERVAL, delta=1.0)   # a third
            self.assertEqual((start + (length - 1) / 2 - 35.5) % self.INTERVAL, 0.0)
        self.assertLess(self._lines(0.25, 0.0).sum(), self._lines(0.5, 0.0).sum())

    def test_the_sine_and_the_triangle_grow_from_the_centre_and_the_saw_from_the_edge(self) -> None:
        crest = 35.5 + self.INTERVAL                                    # the crest at x = 75.5
        for waveform in (Waveform.SINE, Waveform.TRIANGLE):
            start, length = [r for r in self._inner(self._lines(0.5, 0.0, waveform)) if r[0] > 60][0]
            self.assertAlmostEqual(start + (length - 1) / 2, crest)
        start, length = [r for r in self._inner(self._lines(0.5, 0.0, Waveform.SAW)) if r[0] > 60][0]
        self.assertEqual(start, 76)                                     # from the crest outward
        self.assertEqual(length, 10)                                    # a quarter, linear

    def test_the_harmonic_alone_out_is_cutoff_sub_lines_per_interval(self) -> None:
        for cutoff in (2, 3, 4):
            with self.subTest(cutoff=cutoff):
                inner = self._inner(self._lines(0.0, 1.0, cutoff=cutoff))
                base = self._inner(self._lines(1.0, 0.0))
                self.assertAlmostEqual(len(inner), cutoff * len(base), delta=cutoff)
                for _, length in inner:                                 # half the sub-interval, ±1 px of sampling
                    self.assertAlmostEqual(length, 20 / cutoff, delta=1.0)

    def test_a_positive_phase_moves_the_lines_outward(self) -> None:
        base = self._inner(self._lines(1.0, 0.0))
        moved = self._inner(self._lines(1.0, 0.0, phase=self.PHASE + 0.25))
        self.assertEqual(moved[0][0], base[0][0] + 10)

    def test_the_overtone_phase_reshapes_the_wave(self) -> None:
        stacked = self._lines(0.5, 0.5)                                 # overtone in phase: a third
        offset = self._lines(0.5, 0.5, overtone_phase=0.5)              # against: two thirds
        self.assertEqual({length for _, length in self._inner(stacked)}, {14})
        self.assertEqual({length for _, length in self._inner(offset)}, {26})


class MorphologyTest(unittest.TestCase):
    N = 200

    def test_fill_gaps_closes_only_narrow_gaps(self) -> None:
        m = mask(self.N, (10, 50), (69, 100), (120, 160))    # gaps of 19 and 20
        np.testing.assert_array_equal(LinePattern.fill_gaps(m, 20), mask(self.N, (10, 100), (120, 160)))

    def test_remove_slivers_drops_only_narrow_lines(self) -> None:
        m = mask(self.N, (10, 30), (60, 79), (120, 125))     # lines of 20, 19 and 5
        np.testing.assert_array_equal(LinePattern.remove_slivers(m, 20), mask(self.N, (10, 30)))

    def test_morphology_wraps_around_the_projection(self) -> None:
        across = mask(self.N, (190, 200), (0, 15))           # a 25 px line through index 0
        np.testing.assert_array_equal(LinePattern.remove_slivers(across, 20), across)
        short = mask(self.N, (195, 200), (0, 5))             # a 10 px line through index 0
        self.assertFalse(LinePattern.remove_slivers(short, 20).any())
        gap = mask(self.N, (5, 195))                         # a 10 px gap through index 0
        self.assertTrue(LinePattern.fill_gaps(gap, 20).all())

    def test_visible_leaves_a_legal_pattern_unchanged(self) -> None:
        legal = mask(self.N, (0, 20), (40, 70), (90, 110), (150, 175))
        np.testing.assert_array_equal(LinePattern.visible(legal, 20), legal)

    def test_visible_output_has_no_narrow_feature(self) -> None:
        rng = np.random.default_rng(3)
        lengths = rng.integers(1, 60, size=200)                 # alternating lines and gaps of 1–59 px
        m = np.repeat(np.arange(lengths.size) % 2 == 0, lengths)[:3600]
        out = LinePattern.visible(m, 20)
        self.assertTrue(out.any() and not out.all())
        lit = runs(np.roll(out, -int(out.argmin())))        # starts in a gap: no run wraps
        dark = runs(np.roll(~out, -int(out.argmax())))      # starts on a line: no gap wraps
        self.assertGreater(len(lit), 10)
        self.assertTrue(all(length >= 20 for _, length in lit))
        self.assertTrue(all(length >= 20 for _, length in dark))

    def test_dilate_widens_each_line_both_ways(self) -> None:
        m = mask(self.N, (50, 70))
        np.testing.assert_array_equal(LinePattern.dilate(m, 10, 10), mask(self.N, (40, 80)))


if __name__ == "__main__":
    unittest.main()
