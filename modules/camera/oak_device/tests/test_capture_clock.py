"""The capture time of a depthai message on our clock: now less the message's age on depthai's clock. Runs on either
depthai install (`.venv`, `.venv-oakd`)."""

import time
import unittest
from datetime import timedelta

import depthai as dai

from modules.camera.oak_device import capture_clock
from modules.camera.oak_device.capture_clock import capture_time, smoothed_latency


class FakeMessage:
    """A message captured `age` seconds before it was made, as depthai stamps one."""
    def __init__(self, age: float) -> None:
        self._timestamp: timedelta = capture_clock.clock_now() - timedelta(seconds=age)

    def getTimestamp(self) -> timedelta:
        return self._timestamp


class CaptureTimeTest(unittest.TestCase):
    def test_a_message_is_stamped_its_age_before_it_was_made(self) -> None:
        before: float = time.perf_counter()
        message = FakeMessage(0.030)
        after: float = time.perf_counter()
        time.sleep(0.005)                   # handed over later: the stamp stays the capture's
        stamp: float = capture_time(message)
        self.assertGreaterEqual(stamp, before - 0.030 - 0.001)
        self.assertLessEqual(stamp, after - 0.030 + 0.001)

    def test_a_capture_time_after_now_is_stamped_with_the_arrival_and_logged_once(self) -> None:
        capture_clock._future_capture_warned = False
        message = FakeMessage(-0.040)       # 40 ms in the future
        with self.assertLogs('modules.camera.oak_device.capture_clock', 'WARNING') as logs:
            before: float = time.perf_counter()
            stamp: float = capture_time(message)
            after: float = time.perf_counter()
            capture_time(message)
        self.assertTrue(before <= stamp <= after)
        self.assertEqual(len(logs.records), 1)

    def test_the_first_latency_is_the_reading_and_later_ones_pull_it(self) -> None:
        first: float = smoothed_latency(0.0, FakeMessage(0.020))
        self.assertAlmostEqual(first, 20.0, delta=1.0)
        later: float = smoothed_latency(first, FakeMessage(0.120))
        self.assertAlmostEqual(later, first + capture_clock.LATENCY_SMOOTHING * (120.0 - first), delta=1.0)

    def test_depthai_s_clock_is_the_one_messages_are_stamped_on(self) -> None:
        self.assertIsInstance(dai.Clock.now(), timedelta)


if __name__ == '__main__':
    unittest.main()
