"""The screen recorder's size rules, and its states with a fake ring and a fake clip: no GPU, no ffmpeg."""

import time
import unittest
from threading import Lock
from typing import Callable

import numpy as np

from modules.video import (ScreenRecorder, ScreenRecorderSettings, ScreenRecorderState, ScreenSize, even,
                           frames_for, matching_preset, preset_size)

FPS = 64.0      # ticks/s; 1/64 s and its multiples are exact in binary, so spans compare exactly
TIMEOUT = 5.0   # s a test waits for the recorder's thread


class FakeRing:
    """A ring in host memory: each slot holds the number of the tick whose frame it got."""

    def __init__(self, free: int = 1 << 40, fail_allocate: bool = False) -> None:
        self.allocated = False
        self.width = 0
        self.height = 0
        self.free = free
        self.fail_allocate = fail_allocate
        self.frame_on_screen = 0        # what the app drew into the screen frame this tick
        self._lock = Lock()
        self._slots: list[int] = []
        self._pending: tuple[int, int] | None = None
        self._downloads: list[int] = []
        self.allocations = 0
        self.frees = 0
        self.blocks: list[int] = []     # the slots of every block added

    @property
    def frame_bytes(self) -> int:
        return self.width * self.height * 3 // 2

    @property
    def num_slots(self) -> int:
        with self._lock:
            return len(self._slots)

    def allocate(self, width: int, height: int) -> None:
        if self.fail_allocate:
            raise OSError('no CUDA')
        self.allocated, self.width, self.height = True, width, height
        self.allocations += 1

    def deallocate(self) -> None:
        self.allocated = False
        self._pending = None

    def capture(self, slot: int) -> None:
        self.flush()
        self._pending = (slot, self.frame_on_screen)

    def flush(self) -> None:
        if self._pending is not None:
            slot, frame = self._pending
            with self._lock:
                self._slots[slot] = frame
            self._pending = None

    def free_memory(self) -> int:
        return self.free - self.num_slots * self.frame_bytes

    def add_block(self, slots: int) -> None:
        with self._lock:
            self._slots.extend([-1] * slots)
        self.blocks.append(slots)

    def free_blocks(self) -> None:
        with self._lock:
            self._slots = []
        self.frees += 1

    def synchronize(self) -> None:
        pass

    def queue_download(self, slot: int) -> None:
        with self._lock:
            self._downloads.append(self._slots[slot])

    def wait_download(self) -> np.ndarray:
        return np.full((self.height * 3 // 2, self.width), self._downloads.pop(0), np.int64)


class FakeClip:
    """Keeps the first value of each frame it is given."""
    clips: list['FakeClip'] = []

    def __init__(self, path: str, fps: float, quality: int) -> None:
        self.path, self.fps, self.quality = path, fps, quality
        self.comment = ''
        self.frames: list[int] = []
        FakeClip.clips.append(self)

    def add(self, image: np.ndarray) -> None:
        self.frames.append(int(image.flat[0]))

    def close(self) -> bool:
        return bool(self.frames)


class SizeRulesTest(unittest.TestCase):
    def test_even_rounds_down(self) -> None:
        self.assertEqual([even(1919), even(1080), even(1), even(0)], [1918, 1080, 2, 2])

    def test_presets_and_portrait(self) -> None:
        self.assertEqual(preset_size(ScreenSize.UHD, False), (3840, 2160))
        self.assertEqual(preset_size(ScreenSize.HD, True), (1080, 1920))

    def test_matching_preset_in_either_orientation(self) -> None:
        self.assertIs(matching_preset(1440, 2560), ScreenSize.QHD)
        self.assertIs(matching_preset(2560, 1440), ScreenSize.QHD)
        self.assertIs(matching_preset(1918, 1080), ScreenSize.CUSTOM)

    def test_frames_for_a_duration(self) -> None:
        self.assertEqual(frames_for(10.0, 60.0, 1), 600)
        self.assertEqual(frames_for(10.0, 60.0, 2), 300)


class SizeSettingsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = ScreenRecorderSettings()
        self.recorder = ScreenRecorder(self.settings, FakeRing(), 'unused')

    def tearDown(self) -> None:
        self.recorder.stop()

    def test_default_is_upright_4k(self) -> None:
        self.assertEqual((self.settings.width, self.settings.height), (2160, 3840))

    def test_a_preset_writes_the_size(self) -> None:
        self.settings.size = ScreenSize.HD
        self.assertEqual((self.settings.width, self.settings.height, self.settings.portrait), (1080, 1920, True))

    def test_portrait_swaps_the_size(self) -> None:
        self.settings.portrait = False
        self.assertEqual((self.settings.width, self.settings.height), (3840, 2160))
        self.assertIs(self.settings.size, ScreenSize.UHD)

    def test_a_size_by_hand_sets_custom_and_portrait(self) -> None:
        self.settings.width = 1000
        self.settings.height = 600
        self.assertIs(self.settings.size, ScreenSize.CUSTOM)
        self.assertFalse(self.settings.portrait)

    def test_a_size_by_hand_that_is_a_preset_sets_it(self) -> None:
        self.settings.width = 1440
        self.settings.height = 2560
        self.assertIs(self.settings.size, ScreenSize.QHD)


class StateTest(unittest.TestCase):
    def setUp(self) -> None:
        FakeClip.clips = []
        self.settings = ScreenRecorderSettings()
        self.settings.width, self.settings.height = 8, 4
        self.ring = FakeRing()
        self.recorder = ScreenRecorder(self.settings, self.ring, 'unused', FakeClip)
        self.recorder.start()
        self.tick_count = 0
        self.interval = 1.0 / FPS       # s between ticks
        self.now = 0.0
        self.times: dict[int, float] = {}

    def tearDown(self) -> None:
        self.recorder.stop()

    def tick(self) -> None:
        """One render tick, `interval` after the last: the screen frame shows the tick's number when it is drawn."""
        self.tick_count += 1
        self.now += self.interval
        self.times[self.tick_count] = self.now
        if self.recorder.capturing:
            self.ring.frame_on_screen = self.tick_count
        self.recorder.update(self.now)

    def span(self, frames: list[int]) -> float:
        """Seconds from the first frame's tick to the last's."""
        return self.times[frames[-1]] - self.times[frames[0]]

    def tick_until(self, done: Callable[[], bool]) -> None:
        deadline = time.perf_counter() + TIMEOUT
        while not done():
            self.assertLess(time.perf_counter(), deadline, 'the recorder did not get there')
            self.tick()
            time.sleep(0.001)

    def ticks(self, count: int) -> None:
        for _ in range(count):
            self.tick()
            time.sleep(0.001)

    def test_start_records_and_stop_encodes_then_frees(self) -> None:
        self.ticks(3)
        self.assertIs(self.settings.state, ScreenRecorderState.IDLE)
        self.assertFalse(self.ring.allocated)
        ScreenRecorderSettings.start.fire(self.settings)
        self.tick()
        self.assertIs(self.settings.state, ScreenRecorderState.RECORDING)
        self.assertTrue(self.ring.allocated)
        self.tick_until(lambda: self.ring.num_slots > 0)
        self.ticks(20)
        ScreenRecorderSettings.stop.fire(self.settings)
        self.tick()
        self.assertIn(self.settings.state, (ScreenRecorderState.ENCODING, ScreenRecorderState.IDLE))
        self.tick_until(lambda: self.settings.state is ScreenRecorderState.IDLE)
        self.assertFalse(self.ring.allocated)
        self.assertEqual(self.ring.num_slots, 0)
        clip = FakeClip.clips[0]
        self.assertGreaterEqual(len(clip.frames), 20)
        self.assertEqual(clip.frames, sorted(clip.frames))
        self.assertEqual(np.diff(clip.frames).tolist(), [1] * (len(clip.frames) - 1))
        self.assertAlmostEqual(clip.fps, FPS, places=3)
        self.assertEqual(clip.quality, 18)

    def test_start_is_refused_while_encoding(self) -> None:
        ScreenRecorderSettings.start.fire(self.settings)
        self.tick_until(lambda: self.ring.num_slots > 0)
        self.ticks(5)
        ScreenRecorderSettings.stop.fire(self.settings)
        self.tick()
        ScreenRecorderSettings.start.fire(self.settings)
        self.tick_until(lambda: self.settings.state is ScreenRecorderState.IDLE)
        self.ticks(3)
        self.assertIs(self.settings.state, ScreenRecorderState.IDLE)
        self.assertEqual(len(FakeClip.clips), 1)

    def test_duration_stops_the_recording(self) -> None:
        self.settings.duration = 0.5
        ScreenRecorderSettings.start.fire(self.settings)
        self.tick_until(lambda: len(FakeClip.clips) == 1 and self.settings.state is ScreenRecorderState.IDLE)
        frames = FakeClip.clips[0].frames
        self.assertEqual(self.span(frames), 0.5)
        self.assertEqual(len(frames), frames_for(0.5, FPS, 1) + 1)

    def test_a_take_slowing_down_still_spans_its_duration(self) -> None:
        self.settings.duration = 0.5
        ScreenRecorderSettings.start.fire(self.settings)
        self.tick_until(lambda: self.ring.num_slots > 0)
        self.ticks(4)
        self.interval = 2.0 / FPS       # the ticks slow to half the rate measured at Start
        self.tick_until(lambda: len(FakeClip.clips) == 1 and self.settings.state is ScreenRecorderState.IDLE)
        frames = FakeClip.clips[0].frames
        self.assertGreaterEqual(self.span(frames), 0.5)
        self.assertLess(self.span(frames), 0.5 + self.interval)     # it stops at the first tick that reaches it
        self.assertLess(len(frames), frames_for(0.5, FPS, 1))

    def test_frame_step_two_records_every_other_tick(self) -> None:
        self.settings.frame_step = 2
        self.settings.duration = 0.5
        ScreenRecorderSettings.start.fire(self.settings)
        self.tick_until(lambda: len(FakeClip.clips) == 1 and self.settings.state is ScreenRecorderState.IDLE)
        frames = FakeClip.clips[0].frames
        self.assertEqual(self.span(frames), 0.5)
        self.assertEqual(len(frames), frames_for(0.5, FPS, 2) + 1)
        self.assertEqual(set(np.diff(frames).tolist()), {2})
        self.assertAlmostEqual(FakeClip.clips[0].fps, FPS / 2, places=3)

    def test_memory_at_the_reserve_stops_the_recording(self) -> None:
        frame = 8 * 4 * 3 // 2
        self.ring.free = int(self.settings.reserve * (1 << 30)) + 150 * frame     # room for two blocks of 64
        ScreenRecorderSettings.start.fire(self.settings)
        self.tick_until(lambda: len(FakeClip.clips) == 1 and self.settings.state is ScreenRecorderState.IDLE)
        self.assertEqual(len(self.ring.blocks), 2)
        self.assertEqual(len(FakeClip.clips[0].frames), sum(self.ring.blocks))     # every slot of the two blocks

    def test_look_back_keeps_the_last_seconds_and_saves_them(self) -> None:
        self.settings.duration = 0.5
        self.settings.look_back = True
        self.tick()
        self.assertIs(self.settings.state, ScreenRecorderState.RECORDING)
        self.ticks(100)     # more than the ring holds
        self.tick_until(lambda: self.tick_count > 200)
        ScreenRecorderSettings.save.fire(self.settings)
        self.tick()
        saved_at = self.tick_count
        self.tick_until(lambda: len(FakeClip.clips) == 1 and self.settings.state is ScreenRecorderState.RECORDING)
        frames = FakeClip.clips[0].frames
        self.assertGreater(self.ring.num_slots, len(frames))    # the ring holds more than `duration`
        self.assertEqual(self.span(frames), 0.5)                # Save trims it to the last 0.5 s by time
        self.assertEqual(np.diff(frames).tolist(), [1] * (len(frames) - 1))
        self.assertEqual(frames[-1], saved_at)      # the Save tick's own frame, flushed out of the pixel buffer
        self.assertTrue(self.ring.allocated)
        self.assertGreater(self.ring.num_slots, 0)

    def test_look_back_off_frees_the_ring(self) -> None:
        self.settings.look_back = True
        self.tick_until(lambda: self.ring.num_slots > 0)
        self.settings.look_back = False
        self.tick_until(lambda: self.settings.state is ScreenRecorderState.IDLE)
        self.assertFalse(self.ring.allocated)
        self.assertEqual(self.ring.num_slots, 0)
        self.assertEqual(FakeClip.clips, [])

    def test_a_size_change_remakes_the_look_back_ring(self) -> None:
        self.settings.look_back = True
        self.tick_until(lambda: self.ring.num_slots > 0)
        self.settings.width = 16
        self.tick_until(lambda: self.ring.width == 16 and self.settings.state is ScreenRecorderState.RECORDING)
        self.assertEqual(self.ring.allocations, 2)

    def test_a_failed_allocation_is_an_error_and_start_tries_again(self) -> None:
        self.ring.fail_allocate = True
        with self.assertLogs('modules.video.screen_recorder', 'ERROR'):
            ScreenRecorderSettings.start.fire(self.settings)
            self.tick()
        self.assertIs(self.settings.state, ScreenRecorderState.ERROR)
        self.tick_until(lambda: self.ring.frees >= 1)     # the thread frees what the failed start left
        self.ticks(3)
        self.assertIs(self.settings.state, ScreenRecorderState.ERROR)
        self.ring.fail_allocate = False
        ScreenRecorderSettings.start.fire(self.settings)
        self.tick_until(lambda: self.settings.state is ScreenRecorderState.RECORDING)

    def test_exit_while_recording_encodes_what_was_copied(self) -> None:
        ScreenRecorderSettings.start.fire(self.settings)
        self.tick_until(lambda: self.ring.num_slots > 0)
        self.ticks(10)
        self.recorder.stop()
        self.assertEqual(len(FakeClip.clips), 1)
        self.assertGreaterEqual(len(FakeClip.clips[0].frames), 9)
        self.assertEqual(self.ring.num_slots, 0)


if __name__ == '__main__':
    unittest.main()
