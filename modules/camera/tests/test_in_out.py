"""The in and out markers: the range check, the file a recording keeps them in, and on a real chunk
what moving one does to playback."""

import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from modules import video
from modules.camera import FrameType, Player, PlayerSettings, make_chunk_name
from modules.camera.player import IN_OUT_FILE, in_range, read_in_out, write_in_out
from modules.camera.recorder import RecState

from .test_recording import depth_frame, recorder, video_frame


class InRangeTest(unittest.TestCase):
    def test_in_is_inside_and_out_is_not(self) -> None:
        self.assertTrue(in_range((0, 10), (0, 10), (0, 20)))
        self.assertTrue(in_range((0, 19), (0, 10), (0, 20)))
        self.assertFalse(in_range((0, 20), (0, 10), (0, 20)))
        self.assertFalse(in_range((0, 9), (0, 10), (0, 20)))

    def test_across_chunks_the_chunk_comes_first(self) -> None:
        self.assertTrue(in_range((1, 500), (0, 900), (2, 5)))
        self.assertFalse(in_range((2, 5), (0, 900), (2, 5)))
        self.assertFalse(in_range((0, 899), (0, 900), (2, 5)))


class MarkerFileTest(unittest.TestCase):
    def test_a_recording_without_a_file_plays_in_full(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(read_in_out(Path(folder)), [0.0, 1.0])

    def test_markers_survive_a_save(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            write_in_out(Path(folder), [0.25, 0.75])
            self.assertEqual(read_in_out(Path(folder)), [0.25, 0.75])

    def test_a_damaged_file_plays_in_full_and_says_so(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / IN_OUT_FILE).write_text('{"in": 0.2', encoding='utf-8')
            with self.assertLogs('modules.camera.player', level='ERROR'):
                self.assertEqual(read_in_out(Path(folder)), [0.0, 1.0])


def record(root: Path, count: int) -> Path:
    """A recording of `count` frames, one chunk, whose depth tells the frame: 1000 + 30 per frame."""
    rec = recorder([FrameType.VIDEO, FrameType.DEPTH], str(root / 'temp'), 'recordings', str(root))
    rec.start()
    rec.settings.start = True
    deadline: float = time.perf_counter() + 10.0
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
    return next((root / 'recordings').glob('*/' + make_chunk_name(0, 0))).parent


def make_player(root: Path, recording: Path) -> Player:
    settings = PlayerSettings()
    settings.video_frame_types = [FrameType.VIDEO, FrameType.DEPTH]
    settings.video_decoder = video.CoderType.iGPU
    settings.video_path = 'recordings'
    settings.fps = 30.0
    settings.folder = recording.name
    return Player(settings, str(root))


def wait_for(condition, seconds: float = 10.0) -> bool:
    deadline: float = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        if condition():
            return True
        time.sleep(0.005)
    return False


@unittest.skipUnless(video.available() and video.encoder_works(video.Encoder.QSV), 'no iGPU encoder')
class PlaybackTest(unittest.TestCase):
    """One recording, played at 30 fps so a test can act in the middle of it."""
    COUNT: int = 90

    @classmethod
    def setUpClass(cls) -> None:
        cls.folder = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)    # a stopped decoder holds its file for a moment
        cls.root = Path(cls.folder.name)
        cls.recording = record(cls.root, cls.COUNT)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.folder.cleanup()

    def setUp(self) -> None:
        (self.recording / IN_OUT_FILE).unlink(missing_ok=True)
        self.frames: list[int] = []
        self.player: Player = make_player(self.root, self.recording)

        def on_frame(_cam: int, kind: FrameType, image: np.ndarray, _readings: object) -> None:
            if kind == FrameType.DEPTH:
                self.frames.append(round((int(np.median(image)) - 1000) / 30))
        self.player.add_frame_callback(on_frame)

    def tearDown(self) -> None:
        self.player.stop()

    def test_markers_come_back_on_load(self) -> None:
        write_in_out(self.recording, [0.5, 1.0])
        self.player.start()
        self.assertTrue(wait_for(lambda: self.frames))
        self.assertEqual(self.player.settings.in_out, [0.5, 1.0])
        self.assertLess(abs(self.frames[0] - self.COUNT // 2), 4)

    def test_a_move_past_the_playhead_keeps_the_readers(self) -> None:
        self.player.start()
        self.assertTrue(wait_for(lambda: len(self.frames) > 10))
        readers = list(self.player.players)
        self.player.settings.in_out = [0.0, 0.5]
        self.assertTrue(wait_for(lambda: self.player.end_offset == self.COUNT // 2))
        time.sleep(0.2)
        self.assertEqual(self.player.players, readers)
        self.assertTrue(all(p.end_frame == self.COUNT // 2 for p in readers))
        self.assertTrue(wait_for(lambda: len(self.frames) > self.COUNT // 2 + 5))
        self.assertEqual(max(self.frames[:self.COUNT // 2 + 5]), self.COUNT // 2 - 1)   # stopped at out, and on from in
        self.assertEqual(read_in_out(self.recording), [0.0, 0.5])

    def test_a_marker_moved_before_the_playhead_jumps_to_in(self) -> None:
        self.player.start()
        self.assertTrue(wait_for(lambda: len(self.frames) > 40))
        readers = list(self.player.players)
        self.player.settings.in_out = [0.0, 0.25]
        self.assertTrue(wait_for(lambda: self.player.players and self.player.players != readers))
        self.assertTrue(wait_for(lambda: self.frames[-1] < 10))


if __name__ == '__main__':
    unittest.main()
