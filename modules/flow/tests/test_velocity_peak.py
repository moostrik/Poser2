"""The peak: the 99th percentile read from the bins' counts."""

import unittest

import numpy as np

from modules.flow import speed_percentile

LOW, HIGH, BINS = 0.01, 10.0, 256
RATIO: float = (HIGH / LOW) ** (1.0 / BINS)     # each bin this much wider than the one before


class TestSpeedPercentile(unittest.TestCase):

    def test_within_its_bin_it_reads_the_share_of_the_bins_count_it_needs(self) -> None:
        counts = np.zeros(BINS)
        counts[100] = 100
        self.assertAlmostEqual(speed_percentile(counts), LOW * RATIO ** 100.99)

    def test_it_reads_the_bin_the_percentile_falls_in(self) -> None:
        counts = np.zeros(BINS)
        counts[50], counts[200] = 98, 2
        self.assertAlmostEqual(speed_percentile(counts), LOW * RATIO ** 200.5)

    def test_in_the_last_bin_it_reads_the_top(self) -> None:
        counts = np.zeros(BINS)
        counts[BINS - 1] = 100
        self.assertEqual(speed_percentile(counts), HIGH)

    def test_nothing_counted_reads_zero(self) -> None:
        self.assertEqual(speed_percentile(np.zeros(BINS)), 0.0)


if __name__ == '__main__':
    unittest.main()
