"""The capture time of a depthai message on our clock: now less the message's age on depthai's clock.
Runs on a depthai 3 install (`.venv-oak4`)."""

import time
import unittest
from datetime import timedelta
from typing import cast

import depthai as dai

from modules.camera.oak4 import definitions
from modules.camera.oak4.definitions import capture_time


class FakeMessage:
    """A message captured `age` seconds before it was made, as depthai stamps one."""
    def __init__(self, age: float) -> None:
        self._timestamp: timedelta = dai.Clock.now() - timedelta(seconds=age)

    def getTimestamp(self) -> timedelta:
        return self._timestamp


class CaptureTimeTest(unittest.TestCase):
    def test_a_message_is_stamped_its_age_before_it_was_made(self) -> None:
        before: float = time.perf_counter()
        message = FakeMessage(0.030)
        after: float = time.perf_counter()
        time.sleep(0.005)                   # handed over later: the stamp stays the capture's
        stamp: float = capture_time(cast(dai.ImgFrame, message))
        self.assertGreaterEqual(stamp, before - 0.030 - 0.001)
        self.assertLessEqual(stamp, after - 0.030 + 0.001)

    def test_a_capture_time_after_now_is_stamped_with_the_arrival_and_logged_once(self) -> None:
        definitions._future_capture_warned = False
        message = cast(dai.ImgFrame, FakeMessage(-0.040))       # 40 ms in the future
        with self.assertLogs('modules.camera.oak4.definitions', 'WARNING') as logs:
            before: float = time.perf_counter()
            stamp: float = capture_time(message)
            after: float = time.perf_counter()
            capture_time(message)
        self.assertTrue(before <= stamp <= after)
        self.assertEqual(len(logs.records), 1)


if __name__ == '__main__':
    unittest.main()
