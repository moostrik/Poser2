"""A recording with depth: the pairing and the metadata, and the whole way through the real encoders,
the merge and the player where the iGPU is there."""

import math
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from modules import video
from modules.camera import CameraFrame, CameraTracklet, CameraTracklets, ChunkInfo, DepthAlignment, FrameMetadata, FrameReadings, FrameType, Player, \
    PlayerSettings, Recorder, RecorderSettings, TrackingStatus, make_chunk_name, read_metadata
from modules.camera.chunk import frame_line, info_line, parse_metadata
from modules.camera.recorder import RecState
from modules.utils import Rect

LENS: list[list[float]] = [[538.0, 0.0, 450.0], [0.0, 538.0, 360.0], [0.0, 0.0, 1.0]]


def video_frame(i: int) -> CameraFrame:
    return CameraFrame(0, FrameType.VIDEO, np.full((720, 896, 3), i % 255, np.uint8), float(i), readings=FrameReadings(-30.0 + i, 1.0, 1000, 100),
                       intrinsics=LENS, arrival=i + 0.01)


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
        self.recorder.submit_tracklets(tracklets(2.0))          # frame 1's still on their way
        self.assertEqual(self.recorder.frames[0].qsize(), 1)
        self.recorder.submit_tracklets(tracklets(1.0))
        self.assertEqual([self.recorder.frames[0].get_nowait().video.timestamp for _ in range(3)], [0.0, 1.0, 2.0])

    def test_tracklets_that_do_not_come_in_time_are_counted(self) -> None:
        self.recorder.settings.tracklet_wait = 5.0
        self.recorder.submit_tracklets(tracklets(0.0))
        self.recorder.submit_frame(video_frame(0))
        self.recorder.submit_frame(depth_frame(1000, 0.0))
        self.recorder.submit_frame(video_frame(1))
        self.recorder.submit_frame(depth_frame(1000, 1.0))
        time.sleep(0.01)
        self.recorder.sync.update()
        captures = [self.recorder.frames[0].get_nowait() for _ in range(2)]
        self.assertIsNone(captures[1].tracklets)
        self.assertEqual(self.recorder.sync.missed.total('tracklets'), 1)

    def test_a_camera_without_tracklets_is_not_waited_for(self) -> None:
        self.recorder.submit_frame(video_frame(0))
        self.recorder.submit_frame(depth_frame(1000, 0.0))
        self.assertIsNone(self.recorder.frames[0].get_nowait().tracklets)
        self.assertEqual(self.recorder.sync.missed.total('tracklets'), 0)

    def test_the_end_writes_what_waits(self) -> None:
        self.recorder.submit_tracklets(tracklets(0.0))
        self.recorder.submit_frame(video_frame(0))
        self.recorder.submit_frame(depth_frame(1000, 0.0))
        self.recorder.submit_frame(video_frame(1))
        self.recorder.submit_frame(depth_frame(1000, 1.0))
        self.recorder.sync.flush()
        self.assertEqual(self.recorder.frames[0].qsize(), 2)


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
        text = (info_line(info) + frame_line(0, FrameMetadata(1.5, FrameReadings(-30.0, 1.0, 1000, 100), 0.01, [tracklet], 0.033))
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
                rec.submit_frame(video_frame(i))
                rec.submit_frame(depth_frame(1000 + i * 30, float(i)))
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
            player.add_frame_callback(lambda cam, kind, image, frame: played.append((kind, int(np.median(image)), frame.readings if frame else None)))
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


if __name__ == '__main__':
    unittest.main()
