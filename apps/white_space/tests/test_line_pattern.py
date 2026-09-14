"""Tests for LinePattern: the thresholded LFO lines and the legibility morphology."""

import unittest

import numpy as np

from apps.white_space.light.layers import LinePattern


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

    def test_duty_zero_is_dark_and_one_is_solid(self) -> None:
        self.assertFalse(LinePattern.lines(self.X, 40.0, 0.0, 0.0, 2, 0.0, 0.0).any())
        self.assertTrue(LinePattern.lines(self.X, 40.0, 1.0, 0.7, 2, 0.3, 0.2).all())

    PHASE = 0.1125      # line centres at x = 35.5, 75.5, …: every edge falls between two pixels

    def test_half_duty_is_equal_lines_and_gaps(self) -> None:
        lit = LinePattern.lines(self.X, 40.0, 0.5, 0.0, 2, 0.0, self.PHASE)
        inner = [r for r in runs(lit) if r[0] > 0 and r[0] + r[1] < len(lit)]
        self.assertGreater(len(inner), 5)
        for start, length in inner:
            self.assertEqual(length, 20)
            self.assertEqual((start + (length - 1) / 2 - 35.5) % 40.0, 0.0)

    def test_duty_sets_the_line_thickness(self) -> None:
        lit = LinePattern.lines(self.X, 40.0, 0.25, 0.0, 2, 0.0, self.PHASE)
        self.assertEqual({length for start, length in runs(lit) if start > 0 and start + length < 400}, {10})

    def test_full_harmonic_multiplies_the_lines(self) -> None:
        base = runs(LinePattern.lines(self.X, 40.0, 0.5, 0.0, 2, 0.0, self.PHASE))
        doubled = runs(LinePattern.lines(self.X, 40.0, 0.5, 1.0, 2, 0.0, self.PHASE))
        self.assertAlmostEqual(len(doubled), 2 * len(base), delta=1)


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

    def test_legible_leaves_a_legal_pattern_unchanged(self) -> None:
        legal = mask(self.N, (0, 20), (40, 70), (90, 110), (150, 175))
        np.testing.assert_array_equal(LinePattern.legible(legal, 20), legal)

    def test_legible_output_has_no_narrow_feature(self) -> None:
        rng = np.random.default_rng(3)
        lengths = rng.integers(1, 60, size=200)                 # alternating lines and gaps of 1–59 px
        m = np.repeat(np.arange(lengths.size) % 2 == 0, lengths)[:3600]
        out = LinePattern.legible(m, 20)
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
