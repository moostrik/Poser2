"""The time series' slot on the board: each series' samples in a window ending at the newest, from any thread."""

import unittest
from enum import IntEnum
from threading import Thread

import numpy as np

from .. import TimeSeriesStoreMixin


class Lane(IntEnum):
    FIRST = 0
    SECOND = 1


class TimeSeriesStoreTest(unittest.TestCase):
    def test_the_window_ends_at_the_newest_sample_of_any_series(self) -> None:
        store = TimeSeriesStoreMixin()
        store.set_time_sample(Lane.FIRST, 1.0, 10.0)
        store.set_time_sample(Lane.SECOND, 2.0, 14.0)
        store.set_time_sample(Lane.FIRST, 3.0, 13.0)
        end, series = store.get_time_series(2.0)
        self.assertEqual(end, 14.0)
        np.testing.assert_array_equal(series[Lane.FIRST][1], [3.0])     # 10 s is outside 12..14
        np.testing.assert_array_equal(series[Lane.SECOND][1], [2.0])

    def test_samples_from_many_threads_all_arrive(self) -> None:
        store = TimeSeriesStoreMixin()

        def set_samples(offset: int) -> None:
            for i in range(500):
                store.set_time_sample(Lane.FIRST, 1.0, float(offset * 1000 + i))

        threads = [Thread(target=set_samples, args=(n,)) for n in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        _, series = store.get_time_series(1e9)
        self.assertEqual(len(series[Lane.FIRST][0]), 2000)


if __name__ == '__main__':
    unittest.main()
