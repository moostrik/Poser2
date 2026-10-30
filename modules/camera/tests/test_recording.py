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
from modules.camera import CameraFrame, ChunkInfo, FrameReadings, FrameType, Player, PlayerSettings, Recorder, RecorderSettings, \
    make_chunk_name, read_metadata
from modules.camera.chunk import info_line, parse_metadata, readings_line
from modules.camera.recorder import RecState

LENS: list[list[float]] = [[538.0, 0.0, 450.0], [0.0, 538.0, 360.0], [0.0, 0.0, 1.0]]


def video_frame(i: int) -> CameraFrame:
    return CameraFrame(0, FrameType.VIDEO, np.full((720, 896, 3), i % 255, np.uint8), float(i), readings=FrameReadings(-30.0 + i, 1.0, 1000, 100),
                       intrinsics=LENS)


def depth_frame(value: int) -> CameraFrame:
    return CameraFrame(0, FrameType.DEPTH, np.full((300, 374), value, np.uint16), 0.0)


def recorder(types: list[FrameType], temp: str = 'temp', output: str = 'out', data_path: str = '') -> Recorder:
    settings = RecorderSettings()
    settings.video_frame_types = types
    settings.video_encoder = video.CoderType.iGPU
    settings.temp_path = temp
    settings.output_path = output
    settings.fps = 60.0
    settings.keyframe_interval = 1
    return Recorder(settings, data_path)


def stream_frames(path: Path, stream: int) -> int:
    """The frames actually stored in one stream of a chunk."""
    out = subprocess.run(['ffprobe', '-hide_banner', '-loglevel', 'error', '-count_packets', '-select_streams', f'v:{stream}',
                          '-show_entries', 'stream=nb_read_packets', '-of', 'csv=p=0', str(path)], capture_output=True, text=True).stdout
    return int(out.strip().strip(','))


class PairingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.recorder = recorder([FrameType.VIDEO, FrameType.DEPTH])
        self.recorder._set_state(RecState.REC)
        self.recorder.state = RecState.REC

    def tearDown(self) -> None:
        self.recorder.settings.unbind(RecorderSettings.start, self.recorder._on_start)

    def test_a_colour_frame_takes_the_newest_depth(self) -> None:
        self.recorder.submit_frame(depth_frame(1000))
        self.recorder.submit_frame(depth_frame(2000))
        self.recorder.submit_frame(video_frame(0))
        _, depth = self.recorder.frames[0].get_nowait()
        self.assertEqual(int(depth.image[0, 0]), 2000)

    def test_one_depth_frame_serves_every_colour_frame_until_the_next(self) -> None:
        self.recorder.submit_frame(depth_frame(1000))
        self.recorder.submit_frame(video_frame(0))
        self.recorder.submit_frame(video_frame(1))
        self.assertEqual(self.recorder.frames[0].qsize(), 2)

    def test_nothing_is_recorded_before_the_first_depth(self) -> None:
        self.recorder.submit_frame(video_frame(0))
        self.assertTrue(self.recorder.frames[0].empty())


class ColourOnlyTest(unittest.TestCase):
    def test_a_colour_recorder_needs_no_depth_and_keeps_no_metadata(self) -> None:
        colour = recorder([FrameType.VIDEO])
        colour.state = RecState.REC
        colour.submit_frame(video_frame(0))
        frame, depth = colour.frames[0].get_nowait()
        self.assertIsNone(depth)
        self.assertFalse(colour.with_depth)


class MetadataTest(unittest.TestCase):
    def test_the_lines_read_back(self) -> None:
        info = ChunkInfo(900, 720, LENS, True, 60.0)
        text = info_line(info) + readings_line(0, 1.5, FrameReadings(-30.0, 1.0, 1000, 100)) + readings_line(1, 1.6, FrameReadings(math.nan, 1.0, 1000, 100))
        metadata = parse_metadata(text)
        self.assertEqual(metadata.info, info)
        self.assertEqual(metadata.readings[0], FrameReadings(-30.0, 1.0, 1000, 100))
        self.assertTrue(math.isnan(metadata.readings[1].tilt))

    def test_a_line_cut_off_by_a_crash_ends_the_list(self) -> None:
        text = info_line(ChunkInfo(900, 720, LENS, True, 60.0)) + readings_line(0, 1.5, FrameReadings(-30.0, 1.0, 1000, 100)) + '{"frame": 1, "ti'
        self.assertEqual(len(parse_metadata(text).readings), 1)


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
                rec.submit_frame(depth_frame(1000 + i * 30))
                rec.submit_frame(video_frame(i))
                time.sleep(0.002)
            RecorderSettings.split.fire(rec.settings)
            while rec.split_requested.is_set() and time.perf_counter() < deadline:
                time.sleep(0.01)
            time.sleep(0.2)
            for i in range(30, 50):
                rec.submit_frame(depth_frame(1000 + i * 30))
                rec.submit_frame(video_frame(i))
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
                self.assertEqual(len(read_metadata(path).readings), count)
            self.assertEqual(read_metadata(recording / make_chunk_name(0, 1)).readings[0].tilt, 0.0)   # frame 30 opens chunk 1
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
                rec.submit_frame(depth_frame(1000 + i * 30))
                rec.submit_frame(video_frame(i))
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
            self.assertEqual(metadata.readings[3], FrameReadings(-27.0, 1.0, 1000, 100))

            settings = PlayerSettings()
            settings.video_frame_types = [FrameType.VIDEO, FrameType.DEPTH]
            settings.video_decoder = video.CoderType.iGPU
            settings.video_path = 'recordings'
            settings.fps = 120.0
            settings.folder = chunks[0].parent.name
            player = Player(settings, str(root))
            played: list[tuple[FrameType, int, FrameReadings | None]] = []
            player.add_frame_callback(lambda cam, kind, image, readings: played.append((kind, int(np.median(image)), readings)))
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


if __name__ == '__main__':
    unittest.main()
