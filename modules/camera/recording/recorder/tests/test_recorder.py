"""`Recorder`: the bound on what waits for the encoders, the writers, a color-only recorder."""

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from modules import video
from modules.camera import FrameType, RecorderSettings
from modules.camera.recording.recorder.recorder import RecState
from modules.camera.recording.tests.recordings import video_frame, recorder


class QueueTest(unittest.TestCase):
    def setUp(self) -> None:
        self.recorder = recorder([FrameType.VIDEO])
        self.recorder.state = RecState.REC

    def tearDown(self) -> None:
        self.recorder.settings.unbind(RecorderSettings.start, self.recorder._on_start)

    def test_beyond_the_bound_a_capture_is_dropped_whole_and_counted(self) -> None:
        bound: int = 3 * video_frame(0).image.nbytes
        with patch('modules.camera.recording.recorder.recorder.HELD', bound):
            for i in range(5):
                self.recorder.submit_frame(video_frame(i))
        self.assertEqual(self.recorder.frames[0].qsize(), 3)
        self.assertEqual([self.recorder.frames[0].get_nowait().video.timestamp for _ in range(3)], [0.0, 1.0, 2.0])
        self.recorder._update_missed()
        self.assertEqual(self.recorder.settings.dropped, 2)

    def test_a_capture_left_from_the_last_recording_is_not_in_the_next(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            self.recorder.temp_path = Path(folder) / 'temp'
            self.recorder.data_path = folder
            self.recorder.submit_frame(video_frame(0))          # came after the last write
            self.recorder._start_recording()
            try:
                self.assertTrue(self.recorder.frames[0].empty())
            finally:
                self.recorder.state = RecState.STOP
                self.recorder._stop_recording()

    def test_a_start_that_fails_is_given_up_once(self) -> None:
        self.recorder.state = RecState.START
        self.recorder.settings.recording = True
        self.recorder._start_recording = lambda: (_ for _ in ()).throw(OSError('no disk'))  # pyright: ignore[reportAttributeAccessIssue]
        with self.assertLogs('modules.camera.recording.recorder.recorder', 'ERROR'):
            self.recorder._begin()
        self.assertEqual(self.recorder.state, RecState.IDLE)
        self.assertFalse(self.recorder.settings.recording)


class WriterTest(unittest.TestCase):
    def test_a_dead_encoder_ends_the_writer_and_stop_returns(self) -> None:
        writer = video.StreamWriter('no_such_encoder_xyz')
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            with self.assertLogs('modules.video.stream_writer', 'ERROR'):
                writer.start(str(Path(folder) / 'out.mp4'), 30.0)
                deadline = time.perf_counter() + 10.0
                while not writer.failed and time.perf_counter() < deadline:
                    writer.submit_frame(np.zeros((48, 64, 3), np.uint8))
                    time.sleep(0.02)
            self.assertTrue(writer.failed)
            writer.stop()                                       # returns, the encoder gone
            self.assertFalse(writer.thread is not None and writer.thread.is_alive())

    def test_a_stop_right_after_the_start_ends_the_writer(self) -> None:
        writer = video.StreamWriter('libx264')
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            writer.start(str(Path(folder) / 'out.mp4'), 30.0)
            writer.stop()
            self.assertFalse(writer.thread is not None and writer.thread.is_alive())


class ColorOnlyTest(unittest.TestCase):
    def test_a_color_recorder_needs_no_depth(self) -> None:
        color = recorder([FrameType.VIDEO])
        color.state = RecState.REC
        color.submit_frame(video_frame(0))
        self.assertIsNone(color.frames[0].get_nowait().depth)
        self.assertFalse(color.with_depth)


if __name__ == '__main__':
    unittest.main()
