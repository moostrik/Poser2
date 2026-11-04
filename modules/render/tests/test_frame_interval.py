"""The interval the 3D flow divides its motion by: from the capture times, never 0, negative or not finite."""

import math
import unittest

from modules.render.layers.flow.flow_3d_layer import frame_interval

TICK: float = 1.0 / 60.0


class FrameIntervalTest(unittest.TestCase):
    def test_a_capture_gap_is_the_interval(self) -> None:
        self.assertAlmostEqual(frame_interval(1.020, 1.000, 0.020, 2 * TICK, TICK)[0], 0.020)

    def test_the_first_gap_sets_the_period(self) -> None:
        self.assertEqual(frame_interval(1.020, 1.000, None, 2 * TICK, TICK), (1.020 - 1.000, 1.020 - 1.000))

    def test_an_odd_gap_is_held_to_twice_the_period(self) -> None:
        self.assertAlmostEqual(frame_interval(1.100, 1.000, 0.020, 2 * TICK, TICK)[0], 0.040)
        self.assertAlmostEqual(frame_interval(1.002, 1.000, 0.020, 2 * TICK, TICK)[0], 0.010)

    def test_no_frame_gap_takes_the_period_or_the_ticks(self) -> None:
        for timestamp, last in ((None, 1.0), (1.0, None), (math.nan, 1.0), (1.0, math.nan), (1.0, 1.0), (0.9, 1.0), (math.inf, 1.0), (2.0, 1.0)):
            self.assertEqual(frame_interval(timestamp, last, 0.020, 2 * TICK, TICK)[0], 0.020)
            self.assertEqual(frame_interval(timestamp, last, None, 2 * TICK, TICK)[0], 2 * TICK)

    def test_a_bad_period_or_tick_count_falls_back_to_the_time_step(self) -> None:
        for period in (math.nan, 0.0, -1.0, math.inf):
            for ticks in (math.nan, 0.0, -1.0):
                interval, kept = frame_interval(None, None, period, ticks, TICK)
                self.assertEqual((interval, kept), (TICK, None))

    def test_the_interval_is_always_finite_and_positive(self) -> None:
        values = (None, math.nan, math.inf, -math.inf, -1.0, 0.0, 1e-9, 0.02, 1.0)
        for timestamp in values:
            for last in values:
                for period in values:
                    interval = frame_interval(timestamp, last, period, 0.0, TICK)[0]
                    self.assertTrue(math.isfinite(interval) and interval > 0.0, (timestamp, last, period, interval))


if __name__ == '__main__':
    unittest.main()
