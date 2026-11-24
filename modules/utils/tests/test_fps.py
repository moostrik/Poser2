"""Tests for FPS: a rate over the last seconds, from the gaps between arrivals."""

import unittest
from unittest import mock

from modules.utils import FPS


class TestFps(unittest.TestCase):
    def feed(self, fps: FPS, stamps: list[float]) -> None:
        with mock.patch('modules.utils.time.fps.time.perf_counter', side_effect=stamps):
            for _ in stamps:
                fps.processed()

    def test_no_rate_before_two_arrivals(self):
        fps = FPS()
        self.assertEqual(fps.get_rate_average(), 0.0)
        self.feed(fps, [1.0])
        self.assertEqual(fps.get_rate_average(), 0.0)

    def test_the_average_is_the_gaps_over_the_time_they_span(self):
        fps = FPS()
        self.feed(fps, [i / 60.0 for i in range(120)])      # 120 arrivals, 119 gaps of 1/60 s
        self.assertAlmostEqual(fps.get_rate_average(), 60.0)
        self.assertAlmostEqual(fps.get_rate_minimum(), 60.0)
        self.assertAlmostEqual(fps.get_rate_maximum(), 60.0)

    def test_the_window_is_in_seconds_whatever_the_rate(self):
        fps = FPS(window=2.0)
        self.feed(fps, [i / 30.0 for i in range(300)])      # 10 s at 30 fps
        self.assertAlmostEqual(fps.get_rate_average(), 30.0)
        self.assertLessEqual(len(fps._stamps), 2 * 30 + 2)  # about 2 s of arrivals kept, not a count of them

    def test_a_pause_shows_in_the_lowest_rate_of_the_window_after_it(self):
        fps = FPS(window=2.0)
        self.feed(fps, [0.0, 0.1, 0.2, 1.2, 1.3])            # a 1 s gap
        self.assertAlmostEqual(fps.get_rate_minimum(), 1.0)
        self.assertAlmostEqual(fps.get_rate_maximum(), 10.0)
        self.feed(fps, [3.5])                               # the gap reaching into the window still counts
        self.assertAlmostEqual(fps.get_rate_minimum(), 1.0 / 2.2)


if __name__ == '__main__':
    unittest.main()
