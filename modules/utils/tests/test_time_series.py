"""The sample ring and the binning behind the time series plot."""

import unittest
from enum import IntEnum
from threading import Thread

import numpy as np

from modules.board import TimeSeriesStoreMixin
from modules.utils import PerformanceTimer, SampleRing, bin_samples, hold_gaps


class TimerSampleCallbackTest(unittest.TestCase):
    def test_a_callback_hears_each_sample_with_its_name(self) -> None:
        heard: list[tuple[str, float]] = []
        callback = lambda name, time_ms, _timestamp: heard.append((name, time_ms))
        PerformanceTimer.add_sample_callback(callback)
        try:
            timer = PerformanceTimer(name="Probe  ", sample_count=4)
            timer.add_time(2.5, report=False)
            self.assertTrue(PerformanceTimer.has_sample_callbacks())
        finally:
            PerformanceTimer.remove_sample_callback(callback)
        self.assertEqual(heard, [("Probe", 2.5)])
        self.assertFalse(PerformanceTimer.has_sample_callbacks())


class SampleRingTest(unittest.TestCase):
    def test_samples_come_back_oldest_first(self) -> None:
        ring = SampleRing(8)
        for i in range(5):
            ring.add(float(i), i * 10.0)
        timestamps, values = ring.since(0.0)
        np.testing.assert_array_equal(timestamps, [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(values, [0, 10, 20, 30, 40])

    def test_a_full_ring_keeps_the_newest(self) -> None:
        ring = SampleRing(4)
        for i in range(10):
            ring.add(float(i), float(i))
        timestamps, _ = ring.since(0.0)
        np.testing.assert_array_equal(timestamps, [6, 7, 8, 9])
        self.assertEqual(len(ring), 4)

    def test_since_leaves_out_older_samples(self) -> None:
        ring = SampleRing(8)
        for i in range(6):
            ring.add(float(i), float(i))
        timestamps, _ = ring.since(3.5)
        np.testing.assert_array_equal(timestamps, [4, 5])

    def test_an_empty_ring_gives_nothing(self) -> None:
        timestamps, values = SampleRing(4).since(0.0)
        self.assertEqual(len(timestamps), 0)
        self.assertEqual(len(values), 0)

    def test_what_since_returns_is_a_copy(self) -> None:
        ring = SampleRing(4)
        ring.add(1.0, 1.0)
        _, values = ring.since(0.0)
        values[0] = 99.0
        self.assertEqual(ring.since(0.0)[1][0], 1.0)


class BinSamplesTest(unittest.TestCase):
    def test_each_column_holds_its_lowest_and_highest(self) -> None:
        low, high = bin_samples(np.array([0.1, 0.2, 1.5]), np.array([3.0, 5.0, 7.0]), end=2.0, window=2.0, columns=2)
        np.testing.assert_array_equal(low, [3.0, 7.0])
        np.testing.assert_array_equal(high, [5.0, 7.0])

    def test_a_column_without_samples_is_nan(self) -> None:
        low, high = bin_samples(np.array([0.1]), np.array([1.0]), end=2.0, window=2.0, columns=4)
        self.assertTrue(np.isnan(low[1:]).all() and np.isnan(high[1:]).all())

    def test_samples_outside_the_window_are_left_out(self) -> None:
        low, _ = bin_samples(np.array([-1.0, 2.5]), np.array([1.0, 1.0]), end=2.0, window=2.0, columns=2)
        self.assertTrue(np.isnan(low).all())

    def test_gaps_hold_the_last_sample_and_the_start_stays_empty(self) -> None:
        low, high = hold_gaps(np.array([np.nan, 2.0, np.nan, np.nan, 5.0]), np.array([np.nan, 3.0, np.nan, np.nan, 6.0]))
        np.testing.assert_array_equal(low[1:], [2.0, 2.0, 2.0, 5.0])
        np.testing.assert_array_equal(high[1:], [3.0, 3.0, 3.0, 6.0])
        self.assertTrue(np.isnan(low[0]) and np.isnan(high[0]))

    def test_a_spike_survives_many_samples_in_its_column(self) -> None:
        timestamps = np.linspace(0.0, 0.99, 100)
        values = np.ones(100)
        values[50] = 40.0
        _, high = bin_samples(timestamps, values, end=1.0, window=1.0, columns=1)
        self.assertEqual(high[0], 40.0)


class Lane(IntEnum):
    FIRST = 0
    SECOND = 1


class TimeSeriesStoreTest(unittest.TestCase):
    def test_the_window_ends_at_the_newest_sample_of_any_series(self) -> None:
        store = TimeSeriesStoreMixin()
        store.submit_time_sample(Lane.FIRST, 1.0, 10.0)
        store.submit_time_sample(Lane.SECOND, 2.0, 14.0)
        store.submit_time_sample(Lane.FIRST, 3.0, 13.0)
        end, series = store.get_time_series(2.0)
        self.assertEqual(end, 14.0)
        np.testing.assert_array_equal(series[Lane.FIRST][1], [3.0])     # 10 s is outside 12..14
        np.testing.assert_array_equal(series[Lane.SECOND][1], [2.0])

    def test_samples_from_many_threads_all_arrive(self) -> None:
        store = TimeSeriesStoreMixin()

        def submit(offset: int) -> None:
            for i in range(500):
                store.submit_time_sample(Lane.FIRST, 1.0, float(offset * 1000 + i))

        threads = [Thread(target=submit, args=(n,)) for n in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        _, series = store.get_time_series(1e9)
        self.assertEqual(len(series[Lane.FIRST][0]), 2000)


if __name__ == '__main__':
    unittest.main()
