"""The in and out markers: the file a recording keeps them in, and on a real chunk what dragging one
does to playback."""

import subprocess
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from modules import video
from modules.camera import FrameType, Player, PlayerSettings, make_chunk_name
from modules.camera.player import IN_OUT_FILE, read_in_out, write_in_out
from modules.camera.recorder import RecState

from .test_recording import depth_frame, recorder, video_frame


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
    """A recording of `count` frames, one chunk, with keyframes apart as the presets record them. Both
    streams tell the frame: the colour by its grey, the depth as 1000 + 30 per frame."""
    rec = recorder([FrameType.VIDEO, FrameType.DEPTH], str(root / 'temp'), 'recordings', str(root), keyframe_interval=30)
    rec.start()
    rec.settings.start = True
    deadline: float = time.perf_counter() + 10.0
    while rec._get_state() != RecState.REC and time.perf_counter() < deadline:
        time.sleep(0.01)
    for i in range(count):
        rec.submit_frame(video_frame(i, i / 30.0))     # captured at 30 fps: it plays at that rate
        rec.submit_frame(depth_frame(1000 + i * 30, i / 30.0))
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
        self.depths: list[int] = []
        self.colours: list[int] = []
        self.player: Player = make_player(self.root, self.recording)

        def on_frame(_cam: int, kind: FrameType, image: np.ndarray, _metadata: object, _captured: float | None) -> None:
            if kind == FrameType.DEPTH:
                self.depths.append(round((int(np.median(image)) - 1000) / 30))
            else:
                self.colours.append(int(np.median(image)))
        self.player.add_frame_callback(on_frame)

    def tearDown(self) -> None:
        if self.player.is_alive():
            self.player.stop()

    def scrub_to(self, frame: int) -> None:
        """Drag a marker to a frame, and wait for that frame to arrive."""
        self.player.settings.scrub = frame / self.COUNT
        self.assertTrue(wait_for(lambda: self.depths and self.depths[-1] == frame), f'frame {frame}, last {self.depths[-1:]}')

    def test_a_seek_finds_both_streams_whole(self) -> None:
        """A seek goes to a colour keyframe, so depth needs one there too, and its frames stored in
        display order: the encoder calls its forward-only frames B as well, so the type says nothing."""
        chunk = str(self.recording / make_chunk_name(0, 0))

        def packets(stream: int) -> list[tuple[int, bool]]:
            lines = subprocess.run(['ffprobe', '-hide_banner', '-loglevel', 'error', '-select_streams', f'v:{stream}', '-show_entries', 'packet=pts,flags',
                                    '-of', 'csv=p=0', chunk], capture_output=True, text=True).stdout.split()
            return [(int(line.split(',')[0]), 'K' in line) for line in lines]
        colour, depth = packets(0), packets(1)
        self.assertEqual(len(colour), self.COUNT)
        stamps: list[int] = [pts for pts, _key in depth]
        self.assertEqual(stamps, sorted(stamps))
        self.assertEqual(sorted(pts for pts, key in colour if key), [pts for pts, key in depth if key])

    def test_markers_come_back_on_load_and_the_seek_finds_in(self) -> None:
        write_in_out(self.recording, [0.5, 1.0])
        self.player.start()
        self.assertTrue(wait_for(lambda: self.depths))
        self.assertEqual(self.player.settings.in_out, [0.5, 1.0])
        self.assertEqual(self.depths[0], self.COUNT // 2)

    def test_a_drag_pauses_on_the_marker_with_both_streams_of_that_frame(self) -> None:
        self.player.start()
        self.assertTrue(wait_for(lambda: len(self.depths) > 5))
        for frame in (60, 23, 71):
            self.scrub_to(frame)
            self.assertLess(abs(self.colours[-1] - frame), 4)
        arrived: int = len(self.depths)
        time.sleep(0.4)
        self.assertEqual(len(self.depths), arrived)     # one frame each: no playback while paused

    def test_a_drag_keeps_only_the_newest_position(self) -> None:
        self.player.start()
        self.assertTrue(wait_for(lambda: len(self.depths) > 5))
        arrived: int = len(self.depths)
        for frame in range(10, 40):
            self.player.settings.scrub = frame / self.COUNT
        self.assertTrue(wait_for(lambda: self.depths[-1] == 39))
        self.assertLess(len(self.depths) - arrived, 25)

    def test_letting_go_plays_from_in(self) -> None:
        self.player.start()
        self.assertTrue(wait_for(lambda: len(self.depths) > 5))
        self.scrub_to(60)
        self.player.settings.in_out = [1 / 3, 1.0]
        self.player.settings.scrub = -1.0
        self.assertTrue(wait_for(lambda: self.depths[-1] < 60 or self.player.get_current_chunk() < 0, 3.0))
        self.assertTrue(wait_for(lambda: 30 <= self.depths[-1] < 40))
        self.assertEqual(read_in_out(self.recording), [1 / 3, 1.0])

    def test_a_loop_is_told_on_start_and_each_wrap_but_not_on_a_drag(self) -> None:
        loops: list[int] = []
        self.player.add_loop_callback(lambda: loops.append(len(self.depths)))
        write_in_out(self.recording, [0.0, 1 / 3])        # 30 frames: a loop each second
        self.player.start()
        self.assertTrue(wait_for(lambda: len(loops) >= 2, 5.0))
        self.assertEqual(loops[0], 0)
        self.scrub_to(20)
        told: int = len(loops)
        for frame in (5, 25):
            self.scrub_to(frame)
        self.assertEqual(len(loops), told)
        self.player.settings.scrub = -1.0
        self.assertTrue(wait_for(lambda: len(loops) > told))

    def seek_to(self, frame: int) -> None:
        """Press the bar at a frame, as the timeline does, and wait for that frame to arrive."""
        arrived: int = len(self.depths)
        self.player.settings.seek = frame / self.COUNT
        self.player.settings.seek = -1.0
        self.assertTrue(wait_for(lambda: frame in self.depths[arrived:]), f'frame {frame}, last {self.depths[-1:]}')

    def test_pressing_the_bar_plays_on_from_there_and_leaves_the_markers(self) -> None:
        write_in_out(self.recording, [1 / 9, 8 / 9])
        self.player.start()
        self.assertTrue(wait_for(lambda: len(self.depths) > 5))
        arrived: int = len(self.depths)
        self.seek_to(50)
        self.assertTrue(wait_for(lambda: 55 <= self.depths[-1] < 80))      # plays on, in step
        self.assertLess(abs(self.colours[-1] - self.depths[-1]), 4)
        played: list[int] = self.depths[arrived:]
        self.assertEqual(played[played.index(50):], list(range(50, played[-1] + 1)))
        self.assertEqual(self.player.settings.in_out, [1 / 9, 8 / 9])
        self.assertEqual(read_in_out(self.recording), [1 / 9, 8 / 9])

    def test_pressing_the_bar_is_no_loop_but_the_wrap_after_it_is(self) -> None:
        loops: list[int] = []
        self.player.add_loop_callback(lambda: loops.append(len(self.depths)))
        write_in_out(self.recording, [0.0, 1 / 3])
        self.player.start()
        self.assertTrue(wait_for(lambda: len(loops) >= 1 and len(self.depths) > 3))
        told: int = len(loops)
        self.seek_to(5)
        self.assertEqual(len(loops), told)
        self.assertTrue(wait_for(lambda: len(loops) > told, 3.0))
        self.assertTrue(wait_for(lambda: self.depths[-1] < 5))

    def test_pressing_outside_the_markers_shows_the_place_and_plays_from_in(self) -> None:
        write_in_out(self.recording, [1 / 3, 2 / 3])
        self.player.start()
        self.assertTrue(wait_for(lambda: len(self.depths) > 5))
        for frame in (80, 5):
            self.seek_to(frame)
            arrived: int = len(self.depths)
            self.assertTrue(wait_for(lambda: len(self.depths) > arrived + 2 and 30 <= self.depths[-1] < 60, 3.0), f'after {frame}, last {self.depths[-1:]}')
        self.assertEqual(self.player.settings.in_out, [1 / 3, 2 / 3])

    def test_pressing_the_same_place_twice_goes_there_twice(self) -> None:
        self.player.start()
        self.assertTrue(wait_for(lambda: len(self.depths) > 5))
        self.seek_to(40)
        self.assertTrue(wait_for(lambda: self.depths[-1] > 44))
        self.seek_to(40)

    def test_nothing_playing_no_press(self) -> None:
        self.player.start()
        self.assertTrue(wait_for(lambda: self.depths))
        self.player.settings.stop = True
        self.assertTrue(wait_for(lambda: self.player.active_folder == ''))
        self.player.settings.seek = 0.5
        self.assertEqual(self.player.seek_target, 0.0)

    def test_nothing_playing_nothing_paused(self) -> None:
        self.player.start()
        self.assertTrue(wait_for(lambda: self.depths))
        self.player.settings.stop = True
        self.assertTrue(wait_for(lambda: self.player.active_folder == ''))
        self.player.settings.scrub = 0.5
        self.assertEqual(self.player.scrub_target, -1.0)


if __name__ == '__main__':
    unittest.main()
