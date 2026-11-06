"""The interval the flow divides its motion by: the capture gap held around the camera's set frame rate, never 0, negative or not finite."""

import math
import unittest

from modules.render.layers.flow.motion_layer import frame_interval

TICK: float = 1.0 / 60.0
FPS: float = 50.0


class FrameIntervalTest(unittest.TestCase):
    def test_a_capture_gap_is_the_interval(self) -> None:
        self.assertAlmostEqual(frame_interval(1.020, 1.000, FPS, 2 * TICK, TICK), 0.020)
        self.assertAlmostEqual(frame_interval(1.030, 1.000, FPS, 2 * TICK, TICK), 0.030)

    def test_an_odd_gap_is_held_to_the_band_of_the_frame_rate(self) -> None:
        self.assertAlmostEqual(frame_interval(1.100, 1.000, FPS, 2 * TICK, TICK), 0.040)
        self.assertAlmostEqual(frame_interval(1.002, 1.000, FPS, 2 * TICK, TICK), 0.010)

    def test_a_burst_at_startup_does_not_carry_on(self) -> None:
        """A first gap of a few ms is held once, and the next steady gap is taken as it is."""
        timestamps = (0.0, 0.002, 0.022, 0.042, 0.062)
        intervals = [frame_interval(t, last, FPS, TICK, TICK) for last, t in zip(timestamps, timestamps[1:])]
        self.assertAlmostEqual(intervals[0], 0.010)
        for interval in intervals[1:]:
            self.assertAlmostEqual(interval, 0.020)

    def test_no_frame_gap_takes_the_frame_rate(self) -> None:
        for timestamp, last in ((None, 1.0), (1.0, None), (math.nan, 1.0), (1.0, math.nan), (1.0, 1.0), (0.9, 1.0), (math.inf, 1.0), (2.0, 1.0)):
            self.assertAlmostEqual(frame_interval(timestamp, last, FPS, 2 * TICK, TICK), 1.0 / FPS)

    def test_without_a_frame_rate_the_gap_is_taken_as_it_is(self) -> None:
        for fps in (math.nan, 0.0, -1.0, math.inf):
            self.assertAlmostEqual(frame_interval(1.002, 1.000, fps, 2 * TICK, TICK), 0.002)
            self.assertEqual(frame_interval(None, 1.0, fps, 2 * TICK, TICK), 2 * TICK)
            for ticks in (math.nan, 0.0, -1.0):
                self.assertEqual(frame_interval(None, None, fps, ticks, TICK), TICK)

    def test_the_interval_is_always_finite_and_positive(self) -> None:
        values = (None, math.nan, math.inf, -math.inf, -1.0, 0.0, 1e-9, 0.02, 1.0)
        for timestamp in values:
            for last in values:
                for fps in (math.nan, math.inf, -1.0, 0.0, 30.0):
                    interval = frame_interval(timestamp, last, fps, 0.0, TICK)
                    self.assertTrue(math.isfinite(interval) and interval > 0.0, (timestamp, last, fps, interval))


if __name__ == '__main__':
    unittest.main()
