"""The screen recorder's size rules, and its states with a fake ring and a fake clip: no GPU, no ffmpeg."""

import time
import unittest
from threading import Lock
from typing import Callable

import numpy as np

from modules.video import (ScreenMemory, ScreenRecorder, ScreenRecorderSettings, ScreenRecorderState, ScreenSize, even,
                           frames_for, matching_preset, preset_size)

FPS = 64.0      # ticks/s; 1/64 s and its multiples are exact in binary, so spans compare exactly
TIMEOUT = 5.0   # s a test waits for the recorder's thread


class FakeRing:
    """A ring in host memory: each slot holds the number of the tick whose frame it got."""

    def __init__(self, fail_allocate: bool = False) -> None:
        self.allocated = False
        self.width = 0
        self.height = 0
        self.host_memory = True
        self.free: dict[bool, int] = {True: 1 << 40, False: 1 << 40}     # free bytes in RAM (True) and video memory
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

    def allocate(self, width: int, height: int, host_memory: bool) -> None:
        if self.fail_allocate:
            raise OSError('no CUDA')
        self.allocated, self.width, self.height, self.host_memory = True, width, height, host_memory
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

    def free_memory(self, host_memory: bool) -> int:
        held = self.num_slots * self.frame_bytes if host_memory == self.host_memory else 0
        return self.free[host_memory] - held

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

    def save_held(self) -> FakeClip:
        """Wait until the take is held, save the part between in and out, and wait for its file: the take stays held."""
        self.tick_until(lambda: self.settings.state is ScreenRecorderState.HELD)
        saved = len(FakeClip.clips)
        ScreenRecorderSettings.save.fire(self.settings)
        self.tick_until(lambda: len(FakeClip.clips) == saved + 1 and self.settings.state is ScreenRecorderState.HELD)
        return FakeClip.clips[-1]

    def discard(self, then: ScreenRecorderState) -> None:
        ScreenRecorderSettings.discard.fire(self.settings)
        self.tick_until(lambda: self.settings.state is then)

    def test_start_records_and_stop_holds_then_save_and_discard(self) -> None:
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
        self.assertIs(self.settings.state, ScreenRecorderState.HELD)
        self.ticks(5)
        self.assertEqual(FakeClip.clips, [])        # held, not written
        clip = self.save_held()
        self.assertTrue(self.ring.allocated)
        self.assertGreater(self.ring.num_slots, 0)
        self.assertGreaterEqual(len(clip.frames), 20)
        self.assertEqual(np.diff(clip.frames).tolist(), [1] * (len(clip.frames) - 1))
        self.assertAlmostEqual(clip.fps, FPS, places=3)
        self.assertEqual(clip.quality, 18)
        self.discard(ScreenRecorderState.IDLE)
        self.assertFalse(self.ring.allocated)
        self.assertEqual(self.ring.num_slots, 0)
        self.assertEqual(len(FakeClip.clips), 1)

    def test_start_is_refused_while_held(self) -> None:
        ScreenRecorderSettings.start.fire(self.settings)
        self.tick_until(lambda: self.ring.num_slots > 0)
        self.ticks(5)
        ScreenRecorderSettings.stop.fire(self.settings)
        self.tick()
        ScreenRecorderSettings.start.fire(self.settings)
        self.ticks(3)
        self.assertIs(self.settings.state, ScreenRecorderState.HELD)
        self.assertEqual(self.ring.allocations, 1)

    def test_save_writes_between_in_and_out_and_a_second_part(self) -> None:
        self.settings.duration = 1.0
        ScreenRecorderSettings.start.fire(self.settings)
        self.tick_until(lambda: self.settings.state is ScreenRecorderState.HELD)
        whole = self.save_held().frames
        self.settings.in_out = [0.25, 0.75]
        first = self.save_held().frames
        self.assertEqual(self.span(first), 0.5)
        self.assertEqual(self.times[first[0]] - self.times[whole[0]], 0.25)
        self.settings.in_out = [0.75, 1.0]
        second = self.save_held().frames
        self.assertEqual(second[0], first[-1])          # out at 0.75 and in at 0.75 share their frame
        self.assertEqual(second[-1], whole[-1])
        paths = [clip.path for clip in FakeClip.clips]
        self.assertEqual(len(set(paths)), 3)            # three files, none written over
        self.discard(ScreenRecorderState.IDLE)

    def test_the_playhead_follows_the_dragged_marker(self) -> None:
        self.settings.duration = 1.0
        ScreenRecorderSettings.start.fire(self.settings)
        self.tick_until(lambda: self.settings.state is ScreenRecorderState.HELD)
        self.assertEqual(self.settings.playhead[1], 1.0)
        self.settings.scrub = 0.5
        self.tick_until(lambda: self.settings.playhead[0] == 0.5)
        self.settings.scrub = -1.0
        self.ticks(20)
        self.assertEqual(self.settings.playhead[0], 0.5)    # where it was let go
        self.discard(ScreenRecorderState.IDLE)

    def test_the_preview_is_the_frame_under_the_dragged_marker(self) -> None:
        self.settings.duration = 1.0
        ScreenRecorderSettings.start.fire(self.settings)
        self.tick_until(lambda: self.settings.state is ScreenRecorderState.HELD)
        self.assertIsNone(self.recorder.preview_slot)           # no marker dragged
        whole = self.save_held().frames
        self.settings.scrub = 0.5
        slot = self.recorder.preview_slot
        assert slot is not None
        frame = self.ring._slots[slot]
        self.assertEqual(self.times[frame] - self.times[whole[0]], 0.5)     # the frame half a second in
        self.settings.scrub = 1.0
        last = self.recorder.preview_slot
        assert last is not None
        self.assertEqual(self.ring._slots[last], whole[-1])
        self.settings.scrub = -1.0
        self.assertIsNone(self.recorder.preview_slot)           # let go
        self.discard(ScreenRecorderState.IDLE)

    def test_duration_stops_the_recording(self) -> None:
        self.settings.duration = 0.5
        ScreenRecorderSettings.start.fire(self.settings)
        frames = self.save_held().frames
        self.assertEqual(self.span(frames), 0.5)
        self.assertEqual(len(frames), frames_for(0.5, FPS, 1) + 1)

    def test_a_take_slowing_down_still_spans_its_duration(self) -> None:
        self.settings.duration = 0.5
        ScreenRecorderSettings.start.fire(self.settings)
        self.tick_until(lambda: self.ring.num_slots > 0)
        self.ticks(4)
        self.interval = 2.0 / FPS       # the ticks slow to half the rate measured at Start
        frames = self.save_held().frames
        self.assertGreaterEqual(self.span(frames), 0.5)
        self.assertLess(self.span(frames), 0.5 + self.interval)     # it stops at the first tick that reaches it
        self.assertLess(len(frames), frames_for(0.5, FPS, 1))

    def test_frame_step_two_records_every_other_tick(self) -> None:
        self.settings.frame_step = 2
        self.settings.duration = 0.5
        ScreenRecorderSettings.start.fire(self.settings)
        frames = self.save_held().frames
        self.assertEqual(self.span(frames), 0.5)
        self.assertEqual(len(frames), frames_for(0.5, FPS, 2) + 1)
        self.assertEqual(set(np.diff(frames).tolist()), {2})
        self.assertAlmostEqual(FakeClip.clips[0].fps, FPS / 2, places=3)

    def _memory_stops_the_recording(self, memory: ScreenMemory, reserve: float) -> None:
        """Room for two blocks of 64 frames above `reserve` (GB) in `memory`: the take ends with the second."""
        frame = 8 * 4 * 3 // 2
        host_memory = memory is ScreenMemory.HOST
        self.settings.memory = memory
        self.ring.free[host_memory] = int(reserve * (1 << 30)) + 150 * frame
        ScreenRecorderSettings.start.fire(self.settings)
        clip = self.save_held()
        self.assertEqual(self.ring.host_memory, host_memory)
        self.assertEqual(len(self.ring.blocks), 2)
        self.assertEqual(len(clip.frames), sum(self.ring.blocks))     # every slot of the two blocks

    def test_ram_at_its_reserve_stops_the_recording(self) -> None:
        self._memory_stops_the_recording(ScreenMemory.HOST, self.settings.ram_reserve)

    def test_video_memory_at_its_reserve_stops_the_recording(self) -> None:
        self._memory_stops_the_recording(ScreenMemory.VIDEO, self.settings.reserve)

    def test_the_ring_is_in_ram_by_default(self) -> None:
        self.assertIs(self.settings.memory, ScreenMemory.HOST)
        ScreenRecorderSettings.start.fire(self.settings)
        self.tick_until(lambda: self.ring.num_slots > 0)
        self.assertTrue(self.ring.host_memory)

    def test_a_memory_change_remakes_the_look_back_ring(self) -> None:
        self.settings.look_back = True
        self.tick_until(lambda: self.ring.num_slots > 0)
        self.settings.memory = ScreenMemory.VIDEO
        self.tick_until(lambda: not self.ring.host_memory and self.ring.num_slots > 0
                        and self.settings.state is ScreenRecorderState.RECORDING)
        self.assertEqual(self.ring.allocations, 2)

    def test_max_duration_counts_the_memory_set(self) -> None:
        frame = 8 * 4 * 3 // 2
        self.ring.free[True] = int(self.settings.ram_reserve * (1 << 30)) + 640 * frame     # 10 s at 64 ticks/s
        self.ring.free[False] = int(self.settings.reserve * (1 << 30)) + 320 * frame        # 5 s
        # the rate measured settles from 60 to 64 ticks/s, so the estimate lies between the two
        self.tick_until(lambda: 0.0 < self.settings.max_duration < 100.0)
        self.assertTrue(640 / 64 - 0.1 <= self.settings.max_duration <= 640 / 60 + 0.1, self.settings.max_duration)
        self.settings.memory = ScreenMemory.VIDEO
        self.tick_until(lambda: self.settings.max_duration < 7.0)
        self.assertTrue(320 / 64 - 0.1 <= self.settings.max_duration <= 320 / 60 + 0.1, self.settings.max_duration)

    def test_look_back_keeps_the_last_seconds_and_saves_them(self) -> None:
        self.settings.duration = 0.5
        self.settings.look_back = True
        self.tick()
        self.assertIs(self.settings.state, ScreenRecorderState.RECORDING)
        self.ticks(100)     # more than the ring holds
        self.tick_until(lambda: self.tick_count > 200)
        ScreenRecorderSettings.save.fire(self.settings)
        self.tick()
        held_at = self.tick_count
        self.assertIs(self.settings.state, ScreenRecorderState.HELD)
        self.ticks(10)
        self.assertEqual(self.recorder.capturing, False)        # look back pauses while the take is held
        frames = self.save_held().frames
        self.assertGreater(self.ring.num_slots, len(frames))    # the ring holds more than `duration`
        self.assertEqual(self.span(frames), 0.5)                # the hold trims it to the last 0.5 s by time
        self.assertEqual(np.diff(frames).tolist(), [1] * (len(frames) - 1))
        self.assertEqual(frames[-1], held_at)       # the Save tick's own frame, flushed out of the pixel buffer
        slots = self.ring.num_slots
        self.discard(ScreenRecorderState.RECORDING)             # emptied, and recording on in the same ring
        self.assertEqual(self.ring.num_slots, slots)
        self.ticks(5)
        ScreenRecorderSettings.save.fire(self.settings)
        self.tick()
        again = self.save_held().frames
        self.assertGreater(again[0], held_at)       # nothing from before the discard
        self.discard(ScreenRecorderState.RECORDING)

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

    def test_exit_while_held_writes_a_take_never_saved(self) -> None:
        self.settings.duration = 0.5
        ScreenRecorderSettings.start.fire(self.settings)
        self.tick_until(lambda: self.settings.state is ScreenRecorderState.HELD)
        self.recorder.stop()
        self.assertEqual(len(FakeClip.clips), 1)
        self.assertEqual(self.ring.num_slots, 0)

    def test_exit_while_held_after_a_save_writes_nothing_more(self) -> None:
        self.settings.duration = 0.5
        ScreenRecorderSettings.start.fire(self.settings)
        self.save_held()
        self.recorder.stop()
        self.assertEqual(len(FakeClip.clips), 1)
        self.assertEqual(self.ring.num_slots, 0)


if __name__ == '__main__':
    unittest.main()
