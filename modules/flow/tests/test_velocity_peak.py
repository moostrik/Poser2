"""The peak: the speeds' log bins as the shader counts them, and the 99th percentile read from the bins."""

import unittest

import numpy as np

from modules.flow import speed_bin, speed_counts, speed_percentile

LOW, HIGH, BINS = 0.01, 10.0, 256


class TestSpeedBin(unittest.TestCase):

    def test_a_speed_below_the_first_edge_is_still_and_not_counted(self) -> None:
        np.testing.assert_array_equal(speed_bin(np.array([0.0, 0.005, 0.0099])), [-1, -1, -1])
        self.assertEqual(int(speed_counts(np.array([0.0, 0.005])).sum()), 0)

    def test_the_edges_land_in_the_first_and_the_last_bin(self) -> None:
        np.testing.assert_array_equal(speed_bin(np.array([LOW, HIGH * 0.9999, HIGH, 40.0])), [0, BINS - 1, BINS - 1, BINS - 1])

    def test_each_bin_is_the_same_share_wider_than_the_one_before(self) -> None:
        ratio = (HIGH / LOW) ** (1.0 / BINS)
        middle = LOW * ratio ** 100.5      # the middle of bin 100 on the log scale
        self.assertEqual(int(speed_bin(np.array([middle]))[0]), 100)
        self.assertAlmostEqual(ratio - 1.0, 0.027, places=3)


class TestSpeedPercentile(unittest.TestCase):

    def test_it_reads_numpys_percentile_to_within_a_bin(self) -> None:
        speeds = np.exp(np.random.default_rng(5).normal(np.log(0.5), 0.8, 200_000))
        speeds = speeds[(speeds >= LOW) & (speeds < HIGH)]
        expected = float(np.percentile(speeds, 99.0))
        self.assertAlmostEqual(speed_percentile(speed_counts(speeds)) / expected, 1.0, delta=0.027)

    def test_still_pixels_do_not_pull_it_down(self) -> None:
        moving = np.full(1000, 2.0)
        with_still = np.concatenate((moving, np.zeros(100_000)))
        self.assertAlmostEqual(speed_percentile(speed_counts(with_still)), speed_percentile(speed_counts(moving)))

    def test_in_the_last_bin_it_reads_the_top(self) -> None:
        self.assertEqual(speed_percentile(speed_counts(np.full(100, 40.0))), HIGH)

    def test_nothing_counted_reads_zero(self) -> None:
        self.assertEqual(speed_percentile(np.zeros(BINS)), 0.0)


if __name__ == '__main__':
    unittest.main()
