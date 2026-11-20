"""`Recorder`: the bound on what waits for the encoders, the writers, a color-only recorder."""

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import io
import json

import numpy as np

from modules import video
from modules.camera import CameraFrame, CameraTracklet, CameraTracklets, FrameType, TrackingStatus
from modules.recording import ChunkStream, RecorderSettings
from modules.recording.recorder import RecordedCapture
from modules.recording.recorder.recorder import RecState, mirrored_capture
from modules.recording.tests.recordings import LENS, video_frame, recorder
from modules.utils import Rect


class QueueTest(unittest.TestCase):
    def setUp(self) -> None:
        self.recorder = recorder([FrameType.VIDEO])
        self.recorder.state = RecState.REC

    def tearDown(self) -> None:
        self.recorder.settings.unbind(RecorderSettings.start, self.recorder._on_start)

    def test_beyond_the_bound_a_capture_is_dropped_whole_and_counted(self) -> None:
        bound: int = 3 * video_frame(0).image.nbytes
        with patch('modules.recording.recorder.recorder.HELD', bound):
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
        with self.assertLogs('modules.recording.recorder.recorder', 'ERROR'):
            self.recorder._begin()
        self.assertEqual(self.recorder.state, RecState.IDLE)
        self.assertFalse(self.recorder.settings.recording)


class WriterTest(unittest.TestCase):
    def test_a_dead_encoder_ends_the_writer_and_stop_returns(self) -> None:
        writer = video.StreamWriter(video.ClipCodec.H264, video.Hardware.CPU)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            with self.assertLogs('modules.video.stream_writer', 'ERROR'):
                writer.start(str(Path(folder) / 'no_such_folder' / 'out.mp4'), 30.0)      # ffmpeg cannot open it
                deadline = time.perf_counter() + 10.0
                while not writer.failed and time.perf_counter() < deadline:
                    writer.submit_frame(np.zeros((48, 64, 3), np.uint8))
                    time.sleep(0.02)
            self.assertTrue(writer.failed)
            writer.stop()                                       # returns, the encoder gone
            self.assertFalse(writer.thread is not None and writer.thread.is_alive())

    def test_a_stop_right_after_the_start_ends_the_writer(self) -> None:
        writer = video.StreamWriter(video.ClipCodec.H264, video.Hardware.CPU)
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


class _Writer(video.StreamWriter):
    """A writer that keeps what it is handed."""
    def __init__(self) -> None:
        super().__init__(video.ClipCodec.H264, video.Hardware.CPU)
        self.images: list[np.ndarray] = []

    def submit_frame(self, image: np.ndarray, nv12: bool = False) -> None:
        self.images.append(image)


def _half_lit(flip_h: bool) -> RecordedCapture:
    """A capture still to be mirrored or not: its left half lit, one tracklet on the left."""
    image: np.ndarray = np.zeros((4, 8, 3), np.uint8)
    image[:, :4] = 255
    frame = CameraFrame(0, FrameType.VIDEO, image, 1.0, flip_h=flip_h, intrinsics=LENS)
    tracklet = CameraTracklet(cam_id=0, id=1, status=TrackingStatus.TRACKED, roi=Rect(0.1, 0.2, 0.3, 0.4), age=1, timestamp=1.0)
    return RecordedCapture(frame, None, CameraTracklets(0, 1.0, [tracklet]))


class MirroredTest(unittest.TestCase):
    def _written(self, mirrored: bool, flip_h: bool) -> tuple[np.ndarray, list[str]]:
        """What the recorder's thread writes of one capture: the image and the metadata lines."""
        rec = recorder([FrameType.VIDEO])
        rec.settings.mirrored = mirrored
        writer = _Writer()
        rec.recorders[0] = {ChunkStream.VIDEO: writer}
        lines = io.StringIO()
        rec.metadata[0] = lines
        rec.metadata_frames[0] = 0
        rec.chunks[0] = 0
        rec.frames[0].put(_half_lit(flip_h))
        rec._write_queued()
        rec.settings.unbind(RecorderSettings.start, rec._on_start)
        return writer.images[0], lines.getvalue().splitlines()

    def test_on_a_frame_still_to_be_mirrored_is_stored_mirrored_with_its_tracklets_and_flip_h_off(self) -> None:
        image, lines = self._written(mirrored=True, flip_h=True)
        self.assertTrue(np.all(image[:, 4:] == 255) and np.all(image[:, :4] == 0))      # the lit half on the right
        info, frame = json.loads(lines[0]), json.loads(lines[1])
        self.assertFalse(info['flip_h'])
        self.assertAlmostEqual(info['intrinsics'][0][2], 8 - LENS[0][2])                # the lens's center mirrored
        self.assertAlmostEqual(frame['tracklets'][0]['roi'][0], 0.6)                     # 1 - 0.1 - 0.3

    def test_off_a_frame_is_stored_as_the_camera_delivered_it(self) -> None:
        image, lines = self._written(mirrored=False, flip_h=True)
        self.assertTrue(np.all(image[:, :4] == 255))
        self.assertTrue(json.loads(lines[0])['flip_h'])

    def test_on_a_frame_not_to_be_mirrored_is_stored_as_it_is(self) -> None:
        image, lines = self._written(mirrored=True, flip_h=False)
        self.assertTrue(np.all(image[:, :4] == 255))
        self.assertFalse(json.loads(lines[0])['flip_h'])

    def test_the_depth_s_confidence_is_mirrored_with_its_depth(self) -> None:
        confidence: np.ndarray = np.zeros((4, 8), np.uint8)
        confidence[:, :4] = 200
        depth = CameraFrame(0, FrameType.DEPTH, np.zeros((4, 8), np.uint16), 1.0, flip_h=True, confidence=confidence)
        mirrored = mirrored_capture(RecordedCapture(_half_lit(True).video, depth, None))
        assert mirrored.depth is not None and mirrored.depth.confidence is not None
        self.assertTrue(np.all(mirrored.depth.confidence[:, 4:] == 200) and np.all(mirrored.depth.confidence[:, :4] == 0))
        self.assertTrue(np.all(confidence[:, :4] == 200))      # the camera's frame as it was


if __name__ == '__main__':
    unittest.main()
