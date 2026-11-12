"""A recording with depth: the pairing and the metadata, and the whole way through the real encoders,
the merge and the player where the iGPU is there."""

import math
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from typing import Callable
from unittest.mock import patch

import numpy as np

from modules import video
from modules.camera import CameraFrame, CameraTracklet, CameraTracklets, ChunkInfo, DepthAlignment, FrameMetadata, FramePoses, FrameReadings, FrameType, \
    Player, PlayerSettings, Recorder, RecorderSettings, RecordingFolder, TrackingStatus, make_chunk_name, read_metadata
from modules.camera.chunk import frame_line, info_line, parse_metadata
from modules.camera.chunk_readers import ChunkReaders, Decoding
from modules.camera.recorder import RecState
from modules.camera.recording_sync import KEPT
from modules.utils import Rect

LENS: list[list[float]] = [[538.0, 0.0, 450.0], [0.0, 538.0, 360.0], [0.0, 0.0, 1.0]]


def video_frame(i: int, timestamp: float | None = None) -> CameraFrame:
    """Frame `i`, captured at `i` s unless given."""
    t: float = float(i) if timestamp is None else timestamp
    return CameraFrame(0, FrameType.VIDEO, np.full((720, 896, 3), i % 255, np.uint8), t, readings=FrameReadings(-30.0 + i, 1.0, 1000, 100),
                       intrinsics=LENS, arrival=t + 0.01)


POINTS: np.ndarray = np.tile(np.array([[0.25, 0.5, 0.9]], dtype=np.float32), (17, 1))


def poses(timestamp: float) -> FramePoses:
    """The host's poses of camera 0's frame captured at `timestamp`: one player, track 3."""
    return FramePoses(0, timestamp, {3: (np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32), POINTS)}, timestamp + 0.05)


def tracklets(timestamp: float) -> CameraTracklets:
    tracklet = CameraTracklet(cam_id=0, id=1, status=TrackingStatus.TRACKED, roi=Rect(0.1, 0.2, 0.3, 0.4), age=1, timestamp=timestamp)
    return CameraTracklets(0, timestamp, [tracklet], arrival=timestamp + 0.033)


def depth_frame(value: int, timestamp: float) -> CameraFrame:
    return CameraFrame(0, FrameType.DEPTH, np.full((300, 374), value, np.uint16), timestamp)


def recorder(types: list[FrameType], temp: str = 'temp', output: str = 'out', data_path: str = '', keyframe_interval: int = 1) -> Recorder:
    settings = RecorderSettings()
    settings.video_frame_types = types
    settings.video_encoder = video.CoderType.iGPU
    settings.temp_path = temp
    settings.output_path = output
    settings.fps = 60.0
    settings.keyframe_interval = keyframe_interval
    return Recorder(settings, data_path)


def stream_frames(path: Path, stream: int) -> int:
    """The frames actually stored in one stream of a chunk."""
    out = subprocess.run(['ffprobe', '-hide_banner', '-loglevel', 'error', '-count_packets', '-select_streams', f'v:{stream}',
                          '-show_entries', 'stream=nb_read_packets', '-of', 'csv=p=0', str(path)], capture_output=True, text=True).stdout
    return int(out.strip().strip(','))


class PairingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.recorder = recorder([FrameType.VIDEO, FrameType.DEPTH])
        self.recorder.state = RecState.REC

    def tearDown(self) -> None:
        self.recorder.settings.unbind(RecorderSettings.start, self.recorder._on_start)

    def test_a_depth_frame_takes_the_color_frame_captured_with_it(self) -> None:
        for i in range(3):
            self.recorder.submit_frame(video_frame(i))         # color arrives first
        self.recorder.submit_frame(depth_frame(1000, 1.0005))   # within the tolerance of frame 1
        capture = self.recorder.frames[0].get_nowait()
        self.assertEqual(capture.video.timestamp, 1.0)
        assert capture.depth is not None
        self.assertEqual(int(capture.depth.image[0, 0]), 1000)
        self.assertTrue(self.recorder.frames[0].empty())

    def test_a_color_frame_whose_depth_never_comes_is_left_out_and_counted(self) -> None:
        for i in range(3):
            self.recorder.submit_frame(video_frame(i))
        self.recorder.submit_frame(depth_frame(1000, 0.0))      # the first capture
        self.recorder.submit_frame(depth_frame(1000, 2.0))      # frame 1's depth was dropped
        timestamps = [self.recorder.frames[0].get_nowait().video.timestamp for _ in range(2)]
        self.assertEqual(timestamps, [0.0, 2.0])
        self.assertEqual(self.recorder.sync.missed.total('depth'), 1)

    def test_a_color_frame_captured_after_the_depth_waits_for_its_own(self) -> None:
        self.recorder.submit_frame(video_frame(5))
        self.recorder.submit_frame(depth_frame(1000, 4.0))      # no color frame of its own
        self.assertTrue(self.recorder.frames[0].empty())
        self.recorder.submit_frame(depth_frame(2000, 5.0))
        capture = self.recorder.frames[0].get_nowait()
        assert capture.depth is not None
        self.assertEqual((capture.video.timestamp, int(capture.depth.image[0, 0])), (5.0, 2000))

    def test_nothing_is_recorded_before_the_first_depth(self) -> None:
        self.recorder.submit_frame(video_frame(0))
        self.assertTrue(self.recorder.frames[0].empty())

    def test_the_frames_are_kept_as_delivered(self) -> None:
        frame = video_frame(0)
        mirrored = CameraFrame(0, FrameType.VIDEO, frame.image, 0.0, flip_h=True, intrinsics=LENS)
        self.recorder.submit_frame(mirrored)
        self.recorder.submit_frame(depth_frame(1000, 0.0))
        self.assertIs(self.recorder.frames[0].get_nowait().video.image, frame.image)


class TrackletTest(unittest.TestCase):
    def setUp(self) -> None:
        self.recorder = recorder([FrameType.VIDEO, FrameType.DEPTH])
        self.recorder.state = RecState.REC

    def tearDown(self) -> None:
        self.recorder.settings.unbind(RecorderSettings.start, self.recorder._on_start)

    def test_a_capture_waits_for_its_tracklets(self) -> None:
        self.recorder.submit_tracklets(tracklets(0.0))          # this camera sends tracklets
        self.recorder.submit_frame(video_frame(0))
        self.recorder.submit_frame(depth_frame(1000, 0.0))
        self.recorder.submit_frame(video_frame(1))
        self.recorder.submit_frame(depth_frame(1000, 1.0))      # its tracklets are on their way
        self.assertEqual(self.recorder.frames[0].qsize(), 1)
        self.recorder.submit_tracklets(tracklets(1.0))
        captures = [self.recorder.frames[0].get_nowait() for _ in range(2)]
        self.assertEqual([c.tracklets.timestamp if c.tracklets else None for c in captures], [0.0, 1.0])

    def test_a_newer_capture_does_not_end_the_wait_and_the_order_holds(self) -> None:
        self.recorder.submit_tracklets(tracklets(0.0))
        self.recorder.submit_frame(video_frame(0))
        self.recorder.submit_frame(depth_frame(1000, 0.0))
        for i in (1, 2):
            self.recorder.submit_frame(video_frame(i))
            self.recorder.submit_frame(depth_frame(1000, float(i)))
        self.assertEqual(self.recorder.frames[0].qsize(), 1)    # frames 1 and 2 wait, however long
        self.recorder.submit_tracklets(tracklets(1.0))
        self.recorder.submit_tracklets(tracklets(2.0))
        self.assertEqual([self.recorder.frames[0].get_nowait().video.timestamp for _ in range(3)], [0.0, 1.0, 2.0])

    def test_a_later_frame_s_tracklets_settle_a_frame_without_its_own(self) -> None:
        self.recorder.submit_tracklets(tracklets(0.0))
        self.recorder.submit_frame(video_frame(0))
        self.recorder.submit_frame(depth_frame(1000, 0.0))
        self.recorder.submit_frame(video_frame(1))
        self.recorder.submit_frame(depth_frame(1000, 1.0))
        self.recorder.submit_tracklets(tracklets(2.0))          # frame 1's never came
        captures = [self.recorder.frames[0].get_nowait() for _ in range(2)]
        self.assertIsNone(captures[1].tracklets)
        self.assertEqual(self.recorder.sync.missed.total('tracklets'), 1)

    def test_a_stream_that_stopped_holds_no_more_than_kept(self) -> None:
        self.recorder.submit_tracklets(tracklets(0.0))
        for i in range(KEPT + 3):
            self.recorder.submit_frame(video_frame(i))
            self.recorder.submit_frame(depth_frame(1000, float(i)))
        self.assertEqual(self.recorder.sync.waiting, KEPT)

    def test_a_camera_without_tracklets_is_not_waited_for(self) -> None:
        self.recorder.submit_frame(video_frame(0))
        self.recorder.submit_frame(depth_frame(1000, 0.0))
        self.assertIsNone(self.recorder.frames[0].get_nowait().tracklets)
        self.assertEqual(self.recorder.sync.missed.total('tracklets'), 0)

    def test_a_capture_waits_for_the_poses_asked_for(self) -> None:
        self.recorder.expect_poses(0, 0.0, True)
        self.recorder.submit_frame(video_frame(0))
        self.recorder.submit_frame(depth_frame(1000, 0.0))
        self.assertTrue(self.recorder.frames[0].empty())
        self.recorder.submit_poses(poses(0.0))
        capture = self.recorder.frames[0].get_nowait()
        assert capture.poses is not None
        self.assertEqual(list(capture.poses.poses), [3])

    def test_a_frame_nobody_asked_poses_for_does_not_wait(self) -> None:
        self.recorder.expect_poses(0, 0.0, False)
        self.recorder.submit_frame(video_frame(0))
        self.recorder.submit_frame(depth_frame(1000, 0.0))
        self.assertIsNone(self.recorder.frames[0].get_nowait().poses)
        self.assertEqual(self.recorder.sync.missed.total('poses'), 0)

    def test_a_frame_not_yet_said_waits_to_know(self) -> None:
        self.recorder.expect_poses(0, 0.0, False)               # this app says per frame
        self.recorder.submit_frame(video_frame(0))
        self.recorder.submit_frame(depth_frame(1000, 0.0))
        self.recorder.submit_frame(video_frame(1))
        self.recorder.submit_frame(depth_frame(1000, 1.0))      # not said yet for frame 1
        self.assertEqual(self.recorder.frames[0].qsize(), 1)
        self.recorder.expect_poses(0, 1.0, False)
        self.assertEqual(self.recorder.frames[0].qsize(), 2)

    def test_poses_asked_for_and_passed_by_a_later_frame_s_are_counted(self) -> None:
        self.recorder.expect_poses(0, 0.0, True)
        self.recorder.submit_frame(video_frame(0))
        self.recorder.submit_frame(depth_frame(1000, 0.0))
        self.assertTrue(self.recorder.frames[0].empty())
        self.recorder.submit_poses(poses(1.0))                  # the batch of frame 0 was dropped
        self.assertIsNone(self.recorder.frames[0].get_nowait().poses)
        self.assertEqual(self.recorder.sync.missed.total('poses'), 1)

    def test_a_later_frame_s_word_settles_a_frame_none_was_said_of(self) -> None:
        self.recorder.expect_poses(0, 0.0, False)
        self.recorder.submit_frame(video_frame(1))
        self.recorder.submit_frame(depth_frame(1000, 1.0))      # no word of frame 1: its image was dropped
        self.assertTrue(self.recorder.frames[0].empty())
        self.recorder.expect_poses(0, 2.0, False)
        self.assertIsNone(self.recorder.frames[0].get_nowait().poses)

    def test_a_waiting_capture_keeps_its_tracklets_past_kept_newer_ones(self) -> None:
        self.recorder.expect_poses(0, 0.0, True)
        self.recorder.submit_tracklets(tracklets(0.0))
        self.recorder.submit_frame(video_frame(0))
        self.recorder.submit_frame(depth_frame(1000, 0.0))      # waits for its poses
        for i in range(1, KEPT + 2):
            self.recorder.submit_tracklets(tracklets(float(i)))  # its own tracklets pushed out of the stream
        self.recorder.submit_poses(poses(0.0))
        capture = self.recorder.frames[0].get_nowait()
        assert capture.tracklets is not None
        self.assertEqual(capture.tracklets.timestamp, 0.0)

    def test_the_end_writes_what_waits(self) -> None:
        self.recorder.submit_tracklets(tracklets(0.0))
        self.recorder.submit_frame(video_frame(0))
        self.recorder.submit_frame(depth_frame(1000, 0.0))
        self.recorder.submit_frame(video_frame(1))
        self.recorder.submit_frame(depth_frame(1000, 1.0))
        self.recorder.sync.flush()
        self.assertEqual(self.recorder.frames[0].qsize(), 2)


class QueueTest(unittest.TestCase):
    def setUp(self) -> None:
        self.recorder = recorder([FrameType.VIDEO])
        self.recorder.state = RecState.REC

    def tearDown(self) -> None:
        self.recorder.settings.unbind(RecorderSettings.start, self.recorder._on_start)

    def test_beyond_the_bound_a_capture_is_dropped_whole_and_counted(self) -> None:
        bound: int = 3 * video_frame(0).image.nbytes
        with patch('modules.camera.recorder.HELD', bound):
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
        with self.assertLogs('modules.camera.recorder', 'ERROR'):
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


class MetadataTest(unittest.TestCase):
    def test_the_lines_read_back(self) -> None:
        info = ChunkInfo(900, 720, LENS, True, 60.0)
        points = np.array([[0.1, 0.2, 0.9]] * 17, dtype=np.float32)
        tracklet = CameraTracklet(cam_id=2, id=4, status=TrackingStatus.TRACKED, roi=Rect(0.1, 0.2, 0.3, 0.4), age=7, timestamp=1.5, keypoints=points)
        box = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        text = (info_line(info) + frame_line(0, FrameMetadata(1.5, FrameReadings(-30.0, 1.0, 1000, 100), 0.01, [tracklet], 0.033,
                                                              {3: (box, points)}, 0.05))
                + frame_line(1, FrameMetadata(1.6, FrameReadings(math.nan, 1.0, 1000, 100), None, None, None))
                + frame_line(2, FrameMetadata(1.7, None, 0.01, [], 0.03)))
        metadata = parse_metadata(text, cam_id=2)
        self.assertEqual(metadata.info, info)
        first = metadata.frames[0]
        self.assertEqual((first.timestamp, first.readings, first.video_latency, first.tracklet_latency),
                         (1.5, FrameReadings(-30.0, 1.0, 1000, 100), 0.01, 0.033))
        assert first.tracklets is not None and first.tracklets[0].keypoints is not None
        self.assertEqual((first.tracklets[0].cam_id, first.tracklets[0].id, first.tracklets[0].status, first.tracklets[0].roi, first.tracklets[0].age),
                         (2, 4, TrackingStatus.TRACKED, Rect(0.1, 0.2, 0.3, 0.4), 7))
        np.testing.assert_array_almost_equal(first.tracklets[0].keypoints, points)
        assert first.poses is not None
        np.testing.assert_array_almost_equal(first.poses[3][0], box)
        np.testing.assert_array_almost_equal(first.poses[3][1], points)
        self.assertEqual(first.pose_latency, 0.05)
        self.assertIsNone(metadata.frames[1].poses)
        assert metadata.frames[1].readings is not None
        self.assertTrue(math.isnan(metadata.frames[1].readings.tilt))
        self.assertIsNone(metadata.frames[1].tracklets)
        self.assertIsNone(metadata.frames[2].readings)
        self.assertEqual(metadata.frames[2].tracklets, [])

    def test_the_depth_alignment_reads_back(self) -> None:
        alignment = DepthAlignment([[212.4, 0.0, 240.6], [0.0, 212.4, 149.4], [0.0, 0.0, 1.0]], LENS, (896, 720),
                                   [[1.0, 0.0, 0.0, -0.0376], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        info = ChunkInfo(896, 720, LENS, True, 60.0, alignment)
        self.assertEqual(parse_metadata(info_line(info)).info, info)

    def test_a_line_cut_off_by_a_crash_ends_the_list(self) -> None:
        text = info_line(ChunkInfo(900, 720, LENS, True, 60.0)) + frame_line(0, FrameMetadata(1.5)) + '{"frame": 1, "ti'
        self.assertEqual(len(parse_metadata(text).frames), 1)


@unittest.skipUnless(video.available() and video.encoder_works(video.Encoder.QSV), 'no iGPU encoder')
class RoundTripTest(unittest.TestCase):
    def test_a_split_keeps_the_streams_in_step_across_chunks(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            root = Path(folder)
            rec = recorder([FrameType.VIDEO, FrameType.DEPTH], str(root / 'temp'), 'recordings', str(root))
            rec.start()
            rec.settings.start = True
            deadline = time.perf_counter() + 10.0
            while rec._get_state() != RecState.REC and time.perf_counter() < deadline:
                time.sleep(0.01)
            time.sleep(0.1)
            for i in range(30):
                rec.submit_frame(video_frame(i))
                rec.submit_frame(depth_frame(1000 + i * 30, float(i)))
                time.sleep(0.002)
            RecorderSettings.split.fire(rec.settings)
            while rec.split_requested.is_set() and time.perf_counter() < deadline:
                time.sleep(0.01)
            time.sleep(0.2)
            for i in range(30, 50):
                rec.submit_frame(video_frame(i))
                rec.submit_frame(depth_frame(1000 + i * 30, float(i)))
                time.sleep(0.002)
            rec.settings.stop = True
            while rec._get_state() != RecState.IDLE and time.perf_counter() < deadline + 10.0:
                time.sleep(0.01)
            rec.stop()

            recording = next((root / 'recordings').iterdir())
            self.assertEqual(sorted(p.name for p in recording.iterdir()), [make_chunk_name(0, 0), make_chunk_name(0, 1)])
            for chunk, count in ((0, 30), (1, 20)):
                path = recording / make_chunk_name(0, chunk)
                self.assertEqual(stream_frames(path, 0), count)
                self.assertEqual(stream_frames(path, 1), count)
                self.assertEqual(len(read_metadata(path).frames), count)
            readings = read_metadata(recording / make_chunk_name(0, 1)).frames[0].readings
            assert readings is not None
            self.assertEqual(readings.tilt, 0.0)     # frame 30 opens chunk 1
            self.assertEqual(list((root / 'temp').rglob('*.*')), [])

    def test_a_recording_plays_back_in_step(self) -> None:
        count = 40
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:   # a stopped decoder holds its file for a moment
            root = Path(folder)
            rec = recorder([FrameType.VIDEO, FrameType.DEPTH], str(root / 'temp'), 'recordings', str(root))
            rec.start()
            rec.settings.start = True
            deadline = time.perf_counter() + 5.0
            while rec._get_state() != RecState.REC and time.perf_counter() < deadline:
                time.sleep(0.01)
            for i in range(count):
                rec.submit_frame(video_frame(i, i / 60.0))
                rec.submit_frame(depth_frame(1000 + i * 30, i / 60.0))
                time.sleep(0.002)
            rec.settings.stop = True
            while rec._get_state() != RecState.IDLE and time.perf_counter() < deadline + 10.0:
                time.sleep(0.01)
            rec.stop()

            chunks = list((root / 'recordings').glob('*/' + make_chunk_name(0, 0)))
            self.assertEqual(len(chunks), 1)
            self.assertEqual(sorted(p.name for p in chunks[0].parent.iterdir()), [make_chunk_name(0, 0)])
            self.assertEqual(video.StreamReader.frame_count(str(chunks[0])), count)
            metadata = read_metadata(chunks[0])
            self.assertEqual(metadata.info, ChunkInfo(896, 720, LENS, False, 60.0))
            self.assertEqual(metadata.frames[3].readings, FrameReadings(-27.0, 1.0, 1000, 100))
            self.assertAlmostEqual(metadata.frames[3].video_latency or 0.0, 0.01)

            settings = PlayerSettings()
            settings.video_frame_types = [FrameType.VIDEO, FrameType.DEPTH]
            settings.video_decoder = video.CoderType.iGPU
            settings.video_path = 'recordings'
            settings.fps = 120.0
            settings.folder = chunks[0].parent.name
            player = Player(settings, str(root))
            played: list[tuple[FrameType, int, FrameReadings | None]] = []
            moments: list[tuple[float, float | None]] = []      # when each color frame came, and its capture on the play's clock
            player.add_frame_callback(lambda cam, kind, image, frame, captured: played.append((kind, int(np.median(image)), frame.readings if frame else None)))
            player.add_frame_callback(lambda cam, kind, image, frame, captured: moments.append((time.perf_counter(), captured)) if kind == FrameType.VIDEO else None)
            player.start()
            deadline = time.perf_counter() + 10.0
            while len(played) < 2 * count and time.perf_counter() < deadline:
                time.sleep(0.01)
            player.settings.stop = True
            player.stop()

        depths = [entry for entry in played if entry[0] == FrameType.DEPTH]
        colours = [entry for entry in played if entry[0] == FrameType.VIDEO]
        self.assertGreaterEqual(len(depths), count)
        self.assertEqual(len(depths), len(colours))
        for i in range(count):
            self.assertLess(abs(depths[i][1] - (1000 + i * 30)), 40)
            self.assertEqual(depths[i][2], FrameReadings(-30.0 + i, 1.0, 1000, 100))
        # On the recorded time: 1/60 s apart as recorded, not 1/120 as `fps` says, each 10 ms after its capture
        captures = [captured for _, captured in moments[:count]]
        assert all(c is not None for c in captures)
        steps = np.diff(np.array(captures, dtype=np.float64))
        np.testing.assert_allclose(steps, 1.0 / 60.0, atol=1e-6)
        lags = [came - captured for came, captured in moments[:count] if captured is not None]
        self.assertGreater(float(np.median(lags)), 0.009)
        self.assertLess(float(np.median(lags)), 0.03)

    def test_a_color_only_recording_is_one_chunk_with_its_metadata(self) -> None:
        count = 20
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            root = Path(folder)
            rec = recorder([FrameType.VIDEO], str(root / 'temp'), 'recordings', str(root))
            rec.start()
            rec.settings.start = True
            deadline = time.perf_counter() + 5.0
            while rec._get_state() != RecState.REC and time.perf_counter() < deadline:
                time.sleep(0.01)
            for i in range(count):
                rec.submit_frame(video_frame(i))
                time.sleep(0.002)
            rec.settings.stop = True
            while rec._get_state() != RecState.IDLE and time.perf_counter() < deadline + 10.0:
                time.sleep(0.01)
            rec.stop()

            chunks = list((root / 'recordings').glob('*/' + make_chunk_name(0, 0)))
            self.assertEqual(len(chunks), 1)
            self.assertEqual(stream_frames(chunks[0], 0), count)
            metadata = read_metadata(chunks[0])
            self.assertEqual(metadata.info, ChunkInfo(896, 720, LENS, False, 60.0, None, False))
            self.assertEqual(len(metadata.frames), count)


def cam_frame(cam_id: int, value: int, timestamp: float) -> CameraFrame:
    """A small color frame of a camera, its pixels `value`, captured at `timestamp`."""
    return CameraFrame(cam_id, FrameType.VIDEO, np.full((240, 320, 3), value, np.uint8), timestamp, readings=FrameReadings(0.0, 1.0, 1000, 100),
                       intrinsics=LENS, arrival=timestamp + 0.01)


def cam_tracklets(cam_id: int, timestamp: float) -> CameraTracklets:
    tracklet = CameraTracklet(cam_id=cam_id, id=1, status=TrackingStatus.TRACKED, roi=Rect(0.1, 0.2, 0.3, 0.4), age=1, timestamp=timestamp)
    return CameraTracklets(cam_id, timestamp, [tracklet], arrival=timestamp + 0.033)


def record(root: Path, types: list[FrameType], num_cams: int, feed: Callable[[Recorder], None], name: str = '') -> Path:
    """A recording in `root`'s recordings, of what `feed` hands the recorder; its folder."""
    settings = RecorderSettings()
    settings.num_cams = num_cams
    settings.video_frame_types = types
    settings.video_encoder = video.CoderType.iGPU
    settings.temp_path = str(root / 'temp')
    settings.output_path = 'recordings'
    settings.fps = 60.0
    settings.keyframe_interval = 1
    settings.name = name                # two recordings in one second have their own folders
    rec = Recorder(settings, str(root))
    before = set((root / 'recordings').iterdir()) if (root / 'recordings').is_dir() else set()
    rec.start()
    try:
        rec.settings.start = True
        deadline = time.perf_counter() + 5.0
        while rec._get_state() != RecState.REC and time.perf_counter() < deadline:
            time.sleep(0.01)
        feed(rec)
        rec.settings.stop = True
        while rec._get_state() != RecState.IDLE and time.perf_counter() < deadline + 20.0:
            time.sleep(0.01)
    finally:
        rec.stop()
    folders = set((root / 'recordings').iterdir()) - before
    assert len(folders) == 1
    return folders.pop()


def player(root: Path, num_cams: int) -> Player:
    settings = PlayerSettings()
    settings.num_cams = num_cams
    settings.video_frame_types = [FrameType.VIDEO, FrameType.DEPTH]
    settings.video_decoder = video.CoderType.iGPU
    settings.video_path = 'recordings'
    settings.fps = 120.0
    return Player(settings, str(root))


# What a player hands over: (cam, kind, pixel value, recorded capture, capture on the play's clock, metadata)
Played = tuple[int, FrameType, int, float | None, float | None, FrameMetadata | None]


def collect(play: Player, out: list[Played]) -> None:
    play.add_frame_callback(lambda cam, kind, image, frame, captured: out.append(
        (cam, kind, int(np.median(image[..., 0] if image.ndim == 3 else image)), frame.timestamp if frame else None, captured, frame)))


FPS: float = 60.0
SPLIT: int = 30          # frames per camera before the split
COUNT: int = 60


def two_cameras(rec: Recorder) -> None:
    """Camera 0 with color, depth, tracklets and poses; camera 1 color only, 3 ms after camera 0, its every fifth
    frame lost live. A split after frame `SPLIT - 1`."""
    for i in range(COUNT):
        if i == SPLIT:
            time.sleep(0.3)
            RecorderSettings.split.fire(rec.settings)
            time.sleep(0.3)
        t: float = 100.0 + i / FPS
        rec.submit_frame(cam_frame(0, i * 4 % 250, t))
        rec.submit_frame(CameraFrame(0, FrameType.DEPTH, np.full((120, 160), 1000 + i * 30, np.uint16), t))
        rec.submit_tracklets(cam_tracklets(0, t))
        rec.expect_poses(0, t, True)
        rec.submit_poses(poses(t))
        if i % 5 != 4:
            rec.submit_frame(cam_frame(1, i * 4 % 250, t + 0.003))
        time.sleep(0.003)


@unittest.skipUnless(video.available() and video.encoder_works(video.Encoder.QSV), 'no iGPU encoder')
class SeveralCamerasTest(unittest.TestCase):
    """Two cameras with unlike data recorded and played back through the real encoders."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.folder = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.root = Path(cls.folder.name)
        cls.recording = record(cls.root, [FrameType.VIDEO, FrameType.DEPTH], 2, two_cameras)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.folder.cleanup()

    def test_the_chunks_are_cut_at_one_capture_time(self) -> None:
        names = sorted(p.name for p in self.recording.iterdir() if p.suffix == '.mkv')
        self.assertEqual(names, [make_chunk_name(c, k) for c in (0, 1) for k in (0, 1)])
        lost = lambda lo, hi: sum(1 for i in range(lo, hi) if i % 5 == 4)
        for cam, chunk, count, depth in ((0, 0, SPLIT, True), (0, 1, COUNT - SPLIT, True),
                                         (1, 0, SPLIT - lost(0, SPLIT), False), (1, 1, COUNT - SPLIT - lost(SPLIT, COUNT), False)):
            path = self.recording / make_chunk_name(cam, chunk)
            metadata = read_metadata(path, cam)
            assert metadata.info is not None
            self.assertEqual(metadata.info.depth, depth)
            self.assertEqual(len(metadata.frames), count, (cam, chunk))
            self.assertEqual(stream_frames(path, 0), count)
            if depth:
                self.assertEqual(stream_frames(path, 1), count)
        cut: float = 100.0 + (SPLIT - 0.5) / FPS
        for cam in (0, 1):
            self.assertLess(read_metadata(self.recording / make_chunk_name(cam, 0), cam).frames[-1].timestamp, cut)
            self.assertGreater(read_metadata(self.recording / make_chunk_name(cam, 1), cam).frames[0].timestamp, cut)

    def test_both_play_on_the_recorded_time_with_their_own_streams(self) -> None:
        play = player(self.root, 2)
        play.settings.folder = self.recording.name
        played: list[Played] = []
        collect(play, played)
        play.start()
        try:
            deadline = time.perf_counter() + 10.0
            expected: int = 2 * COUNT + COUNT - (COUNT // 5)
            while len(played) < expected and time.perf_counter() < deadline:
                time.sleep(0.01)
        finally:
            play.settings.stop = True
            play.stop()

        def first_loop(cam: int, kind: FrameType) -> list[Played]:
            """A camera's frames of a kind up to where the loop starts again: its recorded time going back."""
            own = [p for p in played if p[0] == cam and p[1] == kind and p[3] is not None]
            return own[:next((i for i in range(1, len(own)) if (own[i][3] or 0.0) < (own[i - 1][3] or 0.0)), len(own))]

        colors = {cam: first_loop(cam, FrameType.VIDEO) for cam in (0, 1)}
        self.assertEqual(len(colors[0]), COUNT)
        self.assertEqual(len(colors[1]), COUNT - COUNT // 5)
        self.assertFalse([p for p in played if p[0] == 1 and p[1] == FrameType.DEPTH])
        self.assertEqual(len(first_loop(0, FrameType.DEPTH)), len(colors[0]))
        for p in colors[0][1:]:     # the first frame went out before a tracklet said the camera sends them
            assert p[5] is not None and p[5].tracklets is not None and p[5].poses is not None
        for p in colors[1]:
            assert p[5] is not None
            self.assertIsNone(p[5].tracklets)
            self.assertIsNone(p[5].poses)
        # One origin for both cameras across the split: each capture on the play's clock is its recorded one moved
        # by the same amount, a catch-up would move it; the first frames are left out, where the decoders start
        offsets = np.array([p[4] - p[3] for cam in (0, 1) for p in colors[cam]
                            if p[3] is not None and p[4] is not None and p[3] > 100.0 + 5.0 / FPS])
        self.assertLess(float(offsets.max() - offsets.min()), 0.002)
        for cam in (0, 1):
            recorded = [p[3] for p in colors[cam]]
            self.assertEqual(recorded, sorted(recorded))

    def test_a_seek_lands_both_cameras_at_one_recorded_time(self) -> None:
        folder = RecordingFolder.find(self.root / 'recordings')[self.recording.name]
        decoding = Decoding(False, False, *video.hwaccel(video.CoderType.iGPU), FPS, '')
        for lo in (3, 4, 9):
            readers = ChunkReaders(folder, 1, lo, None, 2, [FrameType.VIDEO, FrameType.DEPTH], lambda *_: None, decoding)
            try:
                reference = readers.metadata[0].frames[lo].timestamp
                first, _ = readers._range(1, lo, None)
                landed = readers.metadata[1].frames[first].timestamp
                self.assertGreaterEqual(landed, reference - 0.5 / FPS)     # its frame of the same round, or the next one where that was lost
                self.assertLess(landed - reference, 1.5 / FPS)
                self.assertEqual(readers.types, {0: {FrameType.VIDEO, FrameType.DEPTH}, 1: {FrameType.VIDEO}})
            finally:
                readers.stop()
                readers.join()


RIG: int = 3                        # cameras of one rig, as hd_trio's
OFFSETS: list[float] = [0.0, 0.007, 0.013]  # each camera's capture after camera 0's: the cameras run unsynced
LOST: list[int] = [7, 11, 13]       # each camera loses every so many frames live, unlike the others


def rig(with_depth: bool) -> Callable[[Recorder], None]:
    """Every camera of a rig with the same data: color, tracklets and poses, and depth for all or none. A split after
    frame `SPLIT - 1`."""
    def feed(rec: Recorder) -> None:
        for i in range(COUNT):
            if i == SPLIT:
                time.sleep(0.3)
                RecorderSettings.split.fire(rec.settings)
                time.sleep(0.3)
            for cam in range(RIG):
                if i % LOST[cam] == LOST[cam] - 1:
                    continue
                t: float = 100.0 + i / FPS + OFFSETS[cam]
                rec.submit_frame(cam_frame(cam, (i * 4 + cam * 60) % 250, t))
                if with_depth:
                    rec.submit_frame(CameraFrame(cam, FrameType.DEPTH, np.full((120, 160), 1000 + i * 30, np.uint16), t))
                rec.submit_tracklets(cam_tracklets(cam, t))
                rec.expect_poses(cam, t, True)
                rec.submit_poses(FramePoses(cam, t, {3: (np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32), POINTS)}, t + 0.05))
            time.sleep(0.004)
    return feed


@unittest.skipUnless(video.available() and video.encoder_works(video.Encoder.QSV), 'no iGPU encoder')
class RigTest(unittest.TestCase):
    """The cameras of one rig, as hd_trio's and white_space's: one folder, every camera with the same data, recorded
    and played back together in step."""

    def check(self, with_depth: bool) -> None:
        kept = lambda cam, lo, hi: sum(1 for i in range(lo, hi) if i % LOST[cam] != LOST[cam] - 1)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            root = Path(folder)
            types = [FrameType.VIDEO, FrameType.DEPTH] if with_depth else [FrameType.VIDEO]
            recording = record(root, types, RIG, rig(with_depth))

            names = sorted(p.name for p in recording.iterdir() if p.suffix == '.mkv')
            self.assertEqual(names, sorted(make_chunk_name(c, k) for c in range(RIG) for k in (0, 1)))
            cut: float = 100.0 + (SPLIT - 0.5) / FPS
            for cam in range(RIG):
                for chunk, (lo, hi) in enumerate(((0, SPLIT), (SPLIT, COUNT))):
                    path = recording / make_chunk_name(cam, chunk)
                    metadata = read_metadata(path, cam)
                    assert metadata.info is not None
                    self.assertEqual(metadata.info.depth, with_depth)
                    self.assertEqual(len(metadata.frames), kept(cam, lo, hi), (cam, chunk))
                    self.assertEqual(stream_frames(path, 0), len(metadata.frames))
                    if with_depth:
                        self.assertEqual(stream_frames(path, 1), len(metadata.frames))
                    for frame in metadata.frames[1:]:       # the first went out before its tracklets said they come
                        self.assertIsNotNone(frame.tracklets)
                        self.assertIsNotNone(frame.poses)
                    # one cut for the rig: every camera's chunk 0 ends and chunk 1 starts at the same moment
                    if chunk == 0:
                        self.assertLess(metadata.frames[-1].timestamp, cut + OFFSETS[cam])
                    else:
                        self.assertGreater(metadata.frames[0].timestamp, cut)

            play = player(root, RIG)
            play.settings.folder = recording.name
            played: list[Played] = []
            drifts: list[float] = []
            collect(play, played)
            play.add_frame_callback(lambda cam, kind, image, frame, captured: drifts.append(play.get_drift()))
            play.start()
            try:
                deadline = time.perf_counter() + 10.0
                per_loop: int = sum(kept(c, 0, COUNT) for c in range(RIG)) * (2 if with_depth else 1)
                while len(played) < per_loop + 10 and time.perf_counter() < deadline:
                    time.sleep(0.01)
            finally:
                play.settings.stop = True
                play.stop()

        def first_loop(cam: int, kind: FrameType) -> list[Played]:
            own = [p for p in played if p[0] == cam and p[1] == kind and p[3] is not None]
            return own[:next((i for i in range(1, len(own)) if (own[i][3] or 0.0) < (own[i - 1][3] or 0.0)), len(own))]

        offsets: list[float] = []
        for cam in range(RIG):
            colors = first_loop(cam, FrameType.VIDEO)
            self.assertEqual(len(colors), kept(cam, 0, COUNT), cam)
            self.assertEqual(len(first_loop(cam, FrameType.DEPTH)), len(colors) if with_depth else 0)
            recorded = [(i * 4 + cam * 60) % 250 for i in range(COUNT) if i % LOST[cam] != LOST[cam] - 1]
            self.assertLessEqual(max(abs(p[2] - v) for p, v in zip(colors, recorded)), 2)    # its own frames, in order
            for p in colors[1:]:
                assert p[5] is not None and p[5].tracklets is not None and p[5].poses is not None
            offsets += [p[4] - p[3] for p in colors if p[3] is not None and p[4] is not None and p[3] > 100.0 + 5.0 / FPS]
        # one clock for the rig across the split: every camera's frames at their recorded times on one origin
        self.assertLess(max(offsets) - min(offsets), 0.002)
        # the cameras play side by side: never more than a frame and a camera's offset apart
        self.assertLess(float(np.median(drifts)), 1000.0 / FPS + 13.0 + 2.0)

    def test_a_rig_without_depth(self) -> None:
        self.check(False)

    def test_a_rig_with_depth(self) -> None:
        self.check(True)


@unittest.skipUnless(video.available() and video.encoder_works(video.Encoder.QSV), 'no iGPU encoder')
class TwoRecordingsTest(unittest.TestCase):
    def test_the_second_recording_takes_nothing_of_the_first(self) -> None:
        def with_poses(rec: Recorder) -> None:
            for i in range(20):
                t: float = 100.0 + i / FPS
                rec.submit_frame(cam_frame(0, 40, t))
                rec.submit_frame(CameraFrame(0, FrameType.DEPTH, np.full((120, 160), 3000, np.uint16), t))
                rec.expect_poses(0, t, True)
                rec.submit_poses(poses(t))
                time.sleep(0.003)

        def color_only(rec: Recorder) -> None:
            for i in range(20):
                rec.submit_frame(cam_frame(0, 200, 500.0 + i / FPS))
                time.sleep(0.003)

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            root = Path(folder)
            first = record(root, [FrameType.VIDEO, FrameType.DEPTH], 1, with_poses)
            second = record(root, [FrameType.VIDEO], 1, color_only, 'second')
            play = player(root, 1)
            play.settings.folder = first.name
            played: list[Played] = []
            collect(play, played)
            play.start()
            try:
                deadline = time.perf_counter() + 10.0
                while len(played) < 20 and time.perf_counter() < deadline:
                    time.sleep(0.01)
                play.settings.folder = second.name
                while not [p for p in played if p[1] == FrameType.VIDEO and p[2] > 150] and time.perf_counter() < deadline:
                    time.sleep(0.01)
                switched = len(played)
                time.sleep(0.4)
            finally:
                play.settings.stop = True
                play.stop()

        self.assertTrue([p for p in played[:20] if p[1] == FrameType.DEPTH and p[5] is not None and p[5].poses is not None])
        start = next(i for i, p in enumerate(played) if p[1] == FrameType.VIDEO and p[2] > 150)
        after = played[start:]
        self.assertGreater(len(after), 10)
        self.assertLess(start, switched)
        for p in after:
            self.assertEqual(p[1], FrameType.VIDEO)
            self.assertGreater(p[2], 150)
            assert p[5] is not None and p[3] is not None
            self.assertIsNone(p[5].poses)
            self.assertGreaterEqual(p[3], 500.0)


@unittest.skipUnless(video.available() and video.encoder_works(video.Encoder.QSV), 'no iGPU encoder')
class DroppedCaptureTest(unittest.TestCase):
    def test_a_capture_dropped_for_the_bound_keeps_the_streams_and_metadata_alike(self) -> None:
        def flood(rec: Recorder) -> None:
            with patch('modules.camera.recorder.HELD', 4 * (240 * 320 * 3 + 120 * 160 * 2)):
                for i in range(120):
                    t: float = 100.0 + i / FPS
                    rec.submit_frame(cam_frame(0, i * 2 % 250, t))
                    rec.submit_frame(CameraFrame(0, FrameType.DEPTH, np.full((120, 160), 1000 + i * 30, np.uint16), t))
                time.sleep(0.5)

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            recording = record(Path(folder), [FrameType.VIDEO, FrameType.DEPTH], 1, flood)
            path = recording / make_chunk_name(0, 0)
            lines = len(read_metadata(path).frames)
            self.assertLess(lines, 120)             # some were dropped
            self.assertEqual(stream_frames(path, 0), lines)
            self.assertEqual(stream_frames(path, 1), lines)


if __name__ == '__main__':
    unittest.main()
