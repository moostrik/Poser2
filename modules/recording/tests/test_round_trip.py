"""One camera recorded and played back through the real encoders, the merge and the player, where the
iGPU is there: a split, the recorded time, color only, two recordings, captures dropped for the bound."""

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from modules import video
from modules.camera import CameraFrame, FrameReadings, FrameType
from modules.recording import ChunkInfo, Player, PlayerSettings, Recorder, RecorderSettings, make_chunk_name, read_metadata
from modules.recording.recorder.recorder import RecState
from modules.recording.tests.recordings import LENS, video_frame, poses, depth_frame, recorder, stream_frames, stream_images, stream_packets, \
    cam_frame, record, player, Played, clip_metadata, collect, FPS


@unittest.skipUnless(video.available() and video.encoder_works(video.Hardware.iGPU), 'no iGPU encoder')
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
                rec.submit_frame(depth_frame(1000 + i * 30, float(i), 100 + i * 3))
                time.sleep(0.002)
            RecorderSettings.split.fire(rec.settings)
            while rec.split_requested.is_set() and time.perf_counter() < deadline:
                time.sleep(0.01)
            time.sleep(0.2)
            for i in range(30, 50):
                rec.submit_frame(video_frame(i))
                rec.submit_frame(depth_frame(1000 + i * 30, float(i), 100 + i * 3))
                time.sleep(0.002)
            rec.settings.stop = True
            while rec._get_state() != RecState.IDLE and time.perf_counter() < deadline + 10.0:
                time.sleep(0.01)
            rec.stop()

            recording = next((root / 'recordings').iterdir())
            self.assertEqual(sorted(p.name for p in recording.iterdir()), [make_chunk_name(0, 0), make_chunk_name(0, 1)])
            for chunk, count, first in ((0, 30, 0), (1, 20, 30)):
                path = recording / make_chunk_name(0, chunk)
                self.assertEqual(stream_frames(path, 0), count)
                self.assertEqual(stream_frames(path, 1), count)
                self.assertEqual(stream_frames(path, 2), count)
                metadata = read_metadata(path)
                self.assertEqual(len(metadata.frames), count)
                assert metadata.info is not None
                self.assertTrue(metadata.info.confidence)
                confidences = stream_images(path, 2)
                self.assertEqual(len(confidences), count)
                for i, image in enumerate(confidences):       # back within a few levels
                    self.assertLess(abs(float(np.median(image)) - (100 + (first + i) * 3)), 4.0)
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
            self.assertEqual(video.probe(str(chunks[0])).frames, count)
            metadata = read_metadata(chunks[0])
            self.assertEqual(metadata.info, ChunkInfo(896, 720, LENS, False, 60.0))
            self.assertEqual(metadata.frames[3].readings, FrameReadings(-27.0, 1.0, 1000, 100))
            self.assertAlmostEqual(metadata.frames[3].video_latency or 0.0, 0.01)

            settings = PlayerSettings()
            settings.video_frame_types = [FrameType.VIDEO, FrameType.DEPTH]
            settings.video_decoder = video.Hardware.iGPU
            settings.video_path = 'recordings'
            settings.fps = 120.0
            settings.folder = chunks[0].parent.name
            player = Player(settings, str(root))
            played: list[tuple[FrameType, int, FrameReadings | None]] = []
            moments: list[tuple[float, float | None]] = []      # when each color frame came, and its capture on the play's clock
            player.add_frame_callback(lambda clip: played.append((clip.kind, int(np.median(clip.image)),
                                                                  metadata.readings if (metadata := clip_metadata(clip)) else None)))
            player.add_frame_callback(lambda clip: moments.append((time.perf_counter(), clip.timestamp)) if clip.kind == FrameType.VIDEO else None)
            player.start()
            deadline = time.perf_counter() + 10.0
            while len(played) < 2 * count and time.perf_counter() < deadline:
                time.sleep(0.01)
            player.settings.stop = True
            player.stop()

        depths = [entry for entry in played if entry[0] == FrameType.DEPTH]
        colors = [entry for entry in played if entry[0] == FrameType.VIDEO]
        self.assertGreaterEqual(len(depths), count)
        self.assertEqual(len(depths), len(colors))
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


@unittest.skipUnless(video.available() and video.encoder_works(video.Hardware.iGPU), 'no iGPU encoder')
class ConfidencePlaybackTest(unittest.TestCase):
    def test_the_confidence_plays_back_on_its_depth_and_a_seek_finds_all_three_whole(self) -> None:
        """Keyframes apart as the settings files record them: a seek goes to a color keyframe, so the depth and its
        confidence need theirs there. Each stream tells the frame: the depth 1000 + 30 per frame, the confidence
        50 + 2 per frame."""
        count: int = 90
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            root = Path(folder)
            rec = recorder([FrameType.VIDEO, FrameType.DEPTH], str(root / 'temp'), 'recordings', str(root), keyframe_interval=30)
            rec.start()
            rec.settings.start = True
            deadline = time.perf_counter() + 10.0
            while rec._get_state() != RecState.REC and time.perf_counter() < deadline:
                time.sleep(0.01)
            for i in range(count):
                rec.submit_frame(video_frame(i, i / 30.0))
                rec.submit_frame(depth_frame(1000 + i * 30, i / 30.0, 50 + i * 2))
                time.sleep(0.002)
            rec.settings.stop = True
            while rec._get_state() != RecState.IDLE and time.perf_counter() < deadline + 10.0:
                time.sleep(0.01)
            rec.stop()

            chunk = next((root / 'recordings').glob('*/' + make_chunk_name(0, 0)))
            keyframes = [sorted(pts for pts, key in stream_packets(chunk, s) if key) for s in (0, 1, 2)]
            self.assertEqual(keyframes[2], keyframes[0])
            self.assertEqual(keyframes[1], keyframes[0])

            settings = PlayerSettings()
            settings.video_frame_types = [FrameType.VIDEO, FrameType.DEPTH]
            settings.video_decoder = video.Hardware.iGPU
            settings.video_path = 'recordings'
            settings.fps = 30.0
            settings.folder = chunk.parent.name
            play = Player(settings, str(root))
            depths: list[tuple[int, float | None]] = []      # each depth's frame, and its confidence's median
            play.add_frame_callback(lambda clip: depths.append(
                (round((int(np.median(clip.image)) - 1000) / 30), float(np.median(clip.confidence)) if clip.confidence is not None else None))
                if clip.kind == FrameType.DEPTH else None)
            play.start()
            try:
                deadline = time.perf_counter() + 10.0
                while len(depths) < 5 and time.perf_counter() < deadline:
                    time.sleep(0.01)
                arrived: int = len(depths)
                play.settings.seek = 50 / count
                play.settings.seek = -1.0
                while not [d for d in depths[arrived:] if d[0] >= 55] and time.perf_counter() < deadline:
                    time.sleep(0.01)
            finally:
                play.settings.stop = True
                play.stop()

        self.assertTrue(all(c is not None for _, c in depths))
        after: list[tuple[int, float | None]] = depths[arrived:]
        frames: list[int] = [f for f, _ in after]
        self.assertIn(50, frames)
        for f, c in after[frames.index(50):]:
            assert c is not None
            self.assertLess(abs(c - (50 + f * 2)), 4.0, f'frame {f}')


@unittest.skipUnless(video.available() and video.encoder_works(video.Hardware.iGPU), 'no iGPU encoder')
class HevcChunkTest(unittest.TestCase):
    def test_an_hevc_chunk_keeps_its_rate_and_frames(self) -> None:
        """The color part is an mkv whatever the codec: a raw .hevc has no timestamps, and the merge refuses it."""
        count: int = 30

        def color(rec: Recorder) -> None:
            for i in range(count):
                rec.submit_frame(cam_frame(0, 60 + i * 4, 100.0 + i / FPS))
                time.sleep(0.003)

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            recording = record(Path(folder), [FrameType.VIDEO], 1, color, codec=video.ClipCodec.HEVC)
            chunk = recording / make_chunk_name(0, 0)
            self.assertTrue(chunk.is_file())
            info = video.probe(str(chunk))
            self.assertEqual(info.frames, count)
            self.assertAlmostEqual(info.fps, FPS)


@unittest.skipUnless(video.available() and video.encoder_works(video.Hardware.iGPU), 'no iGPU encoder')
class TwoRecordingsTest(unittest.TestCase):
    def test_the_second_recording_takes_nothing_of_the_first(self) -> None:
        def with_poses(rec: Recorder) -> None:
            for i in range(20):
                t: float = 100.0 + i / FPS
                rec.submit_frame(cam_frame(0, 40, t))
                rec.submit_frame(CameraFrame(0, FrameType.DEPTH, np.full((120, 160), 3000, np.uint16), t))
                rec.expect_poses(0, t, True)
                rec.submit_poses(poses(t), t + 0.05)
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


@unittest.skipUnless(video.available() and video.encoder_works(video.Hardware.iGPU), 'no iGPU encoder')
class DroppedCaptureTest(unittest.TestCase):
    def test_a_capture_dropped_for_the_bound_keeps_the_streams_and_metadata_alike(self) -> None:
        def flood(rec: Recorder) -> None:
            with patch('modules.recording.recorder.recorder.HELD', 4 * (240 * 320 * 3 + 120 * 160 * 2)):
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
