"""The recording session: the panel's start reaches the recorder and the timer, the timer's split the recorder."""

import time
import unittest

from apps.deep_flow.settings import Settings
from modules.camera import RecorderSettings
from modules.session import Session, SessionSettings


class SessionWiringTest(unittest.TestCase):
    def setUp(self) -> None:
        self.camera = Settings().camera
        self.starts: list[bool] = []
        self.splits: list[bool] = []
        self.camera.recorder.bind(RecorderSettings.start, self.starts.append)
        self.camera.recorder.bind(RecorderSettings.split, self.splits.append)
        self.session = Session(self.camera.control)

    def tearDown(self) -> None:
        SessionSettings.stop.fire(self.camera.control)

    def test_recordings_split_every_minute_by_default(self) -> None:
        self.assertEqual(self.camera.split_seconds, 60.0)
        self.assertEqual(self.camera.control.split_seconds, 60.0)

    def test_the_timer_splits_the_recorder(self) -> None:
        self.camera.split_seconds = 1
        SessionSettings.start.fire(self.camera.control)
        self.assertEqual(len(self.starts), 1)
        time.sleep(1.2)
        self.assertEqual(len(self.splits), 1)


if __name__ == '__main__':
    unittest.main()
