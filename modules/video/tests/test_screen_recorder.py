"""The screen recorder's size rules, and its states fed plain frames as the capture hands them over: no GPU, no
ffmpeg."""

import time
import unittest
from threading import Lock
from typing import Callable

import numpy as np

from modules.video import BitDepth, Chroma, ScreenFormat, ScreenFrame, ScreenRecorder, ScreenRecorderFrameSettings, \
    ScreenRecorderSettings, ScreenSize
from modules.video.screen.screen_recorder import ScreenRecorderRecordSettings, ScreenRecorderTakeSettings, \
    ScreenRecorderState, frames_for
from modules.video.screen.screen_size import even, matching_preset, preset_size

FPS = 64.0      # ticks/s; 1/64 s and its multiples are exact in binary, so spans compare exactly
TIMEOUT = 5.0   # s a test waits for the recorder's thread
WIDTH, HEIGHT = 8, 4
FRAME_BYTES = ScreenFormat(WIDTH, HEIGHT).frame_bytes      # 4:4:4, the default
GB = 1 << 30


def tick_of(frame: np.ndarray) -> int:
    """The tick number a test frame carries in its first two bytes."""
    return int(frame.flat[0]) | int(frame.flat[1]) << 8


class FakeClip:
    """Keeps the tick number of each frame it is given."""
    clips: list['FakeClip'] = []

    def __init__(self, path: str, fps: float, quality: int, screen_format: ScreenFormat) -> None:
        self.path, self.fps, self.quality, self.format = path, fps, quality, screen_format
        self.comment = ''
        self.frames: list[int] = []
        FakeClip.clips.append(self)

    def add(self, image: np.ndarray) -> None:
        self.frames.append(tick_of(image))

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

    def test_a_format_s_bytes_follow_its_chroma(self) -> None:
        nv12 = ScreenFormat(2160, 3840, Chroma.YUV420)
        full = ScreenFormat(2160, 3840, Chroma.YUV444)
        self.assertEqual(nv12.shape, (3840 * 3 // 2, 2160))
        self.assertEqual(full.shape, (3840 * 3, 2160))
        self.assertEqual(nv12.frame_bytes, 2160 * 3840 * 3 // 2)
        self.assertEqual(full.frame_bytes, 2 * nv12.frame_bytes)

    def test_ten_bits_take_16_bit_samples_or_one_packed_plane(self) -> None:
        half = ScreenFormat(2160, 3840, Chroma.YUV420, BitDepth.BIT_10)
        self.assertEqual((half.shape, half.dtype), (ScreenFormat(2160, 3840, Chroma.YUV420).shape, np.uint16))
        self.assertEqual(half.frame_bytes, 2 * ScreenFormat(2160, 3840, Chroma.YUV420).frame_bytes)
        full = ScreenFormat(2160, 3840, Chroma.YUV444, BitDepth.BIT_10)
        self.assertTrue(full.packed)
        self.assertEqual((full.shape, full.dtype), ((3840, 2160), np.uint32))
        self.assertEqual(full.frame_bytes, 2160 * 3840 * 4)


class SizeSettingsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = ScreenRecorderSettings()
        self.recorder = ScreenRecorder(self.settings, 'unused')

    def tearDown(self) -> None:
        self.recorder.stop()

    def test_default_is_upright_4k(self) -> None:
        self.assertEqual((self.settings.frame.width, self.settings.frame.height), (2160, 3840))

    def test_a_preset_writes_the_size(self) -> None:
        self.settings.frame.size = ScreenSize.HD
        self.assertEqual((self.settings.frame.width, self.settings.frame.height, self.settings.frame.portrait), (1080, 1920, True))

    def test_portrait_swaps_the_size(self) -> None:
        self.settings.frame.portrait = False
        self.assertEqual((self.settings.frame.width, self.settings.frame.height), (3840, 2160))
        self.assertIs(self.settings.frame.size, ScreenSize.UHD)

    def test_a_size_by_hand_sets_custom_and_portrait(self) -> None:
        self.settings.frame.width = 1000
        self.settings.frame.height = 600
        self.assertIs(self.settings.frame.size, ScreenSize.CUSTOM)
        self.assertFalse(self.settings.frame.portrait)

    def test_a_size_by_hand_that_is_a_preset_sets_it(self) -> None:
        self.settings.frame.width = 1440
        self.settings.frame.height = 2560
        self.assertIs(self.settings.frame.size, ScreenSize.QHD)


class StateTest(unittest.TestCase):
    """The test plays the capture: each tick the recorder captures, it hands over a frame carrying the tick's number."""

    def setUp(self) -> None:
        FakeClip.clips = []
        self.settings = ScreenRecorderSettings()
        self.settings.frame.width, self.settings.frame.height = WIDTH, HEIGHT
        self.free = 1 << 40         # bytes of free RAM before the store
        self.recorder = ScreenRecorder(self.settings, 'unused', FakeClip, lambda: self.free - self._store_bytes())
        self.recorder.start()
        self.tick_count = 0
        self.interval = 1.0 / FPS
        self.now = 0.0
        self.times: dict[int, float] = {}
        self.lock = Lock()
        self.lent = 0
        self.handed = 0

    def tearDown(self) -> None:
        self.recorder.stop()

    def _store_bytes(self) -> int:
        return self.recorder._store.nbytes

    def frame(self, tick: int, timestamp: float) -> ScreenFrame:
        """A frame at the size the capture is made at, as a real capture hands it over."""
        screen_format = self.recorder.capture_format or ScreenFormat(WIDTH, HEIGHT)
        data = np.zeros(screen_format.shape, screen_format.dtype)
        data.flat[0], data.flat[1] = tick & 0xFF, tick >> 8
        with self.lock:
            self.lent += 1
            self.handed += 1
        return ScreenFrame(data, timestamp, 0, self._give_back)

    def _give_back(self) -> None:
        with self.lock:
            self.lent -= 1

    def tick(self) -> None:
        """One render tick, `interval` after the last: the recorder's update, then the frame it captures."""
        self.tick_count += 1
        self.now += self.interval
        self.times[self.tick_count] = self.now
        self.recorder.update(self.now)
        if self.recorder.capturing:
            self.recorder.submit_screen_frame(self.frame(self.tick_count, self.now))

    def span(self, frames: list[int]) -> float:
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

    def stored(self) -> int:
        with self.recorder._lock:
            return self.recorder._written

    def save_held(self) -> FakeClip:
        self.tick_until(lambda: self.settings.record.state is ScreenRecorderState.HELD)
        saved = len(FakeClip.clips)
        ScreenRecorderTakeSettings.save.fire(self.settings.take)
        self.tick_until(lambda: len(FakeClip.clips) == saved + 1 and self.settings.record.state is ScreenRecorderState.HELD)
        return FakeClip.clips[-1]

    def discard(self, then: ScreenRecorderState) -> None:
        ScreenRecorderTakeSettings.discard.fire(self.settings.take)
        self.tick_until(lambda: self.settings.record.state is then)

    def test_start_records_and_stop_holds_then_save_and_discard(self) -> None:
        self.ticks(3)
        self.assertIs(self.settings.record.state, ScreenRecorderState.IDLE)
        self.assertIsNone(self.recorder.capture_format)
        ScreenRecorderRecordSettings.start.fire(self.settings.record)
        self.tick()
        self.assertIs(self.settings.record.state, ScreenRecorderState.RECORDING)
        self.assertEqual(self.recorder.capture_format, ScreenFormat(WIDTH, HEIGHT, Chroma.YUV444))
        self.tick_until(lambda: self.stored() >= 20)
        ScreenRecorderRecordSettings.stop.fire(self.settings.record)
        self.tick()
        self.assertIs(self.settings.record.state, ScreenRecorderState.HELD)
        self.ticks(5)
        self.assertEqual(FakeClip.clips, [])        # held, not written
        clip = self.save_held()
        self.assertGreaterEqual(len(clip.frames), 20)
        self.assertEqual(np.diff(clip.frames).tolist(), [1] * (len(clip.frames) - 1))
        self.assertAlmostEqual(clip.fps, FPS, places=3)
        self.assertEqual(clip.quality, 18)
        self.discard(ScreenRecorderState.IDLE)
        self.assertIsNone(self.recorder.capture_format)
        self.assertEqual(self.recorder._store.num_slots, 0)
        self.tick_until(lambda: self.lent == 0)     # every frame given back

    def test_start_is_refused_while_held(self) -> None:
        ScreenRecorderRecordSettings.start.fire(self.settings.record)
        self.tick_until(lambda: self.stored() >= 5)
        ScreenRecorderRecordSettings.stop.fire(self.settings.record)
        self.tick()
        ScreenRecorderRecordSettings.start.fire(self.settings.record)
        self.ticks(3)
        self.assertIs(self.settings.record.state, ScreenRecorderState.HELD)

    def test_save_writes_between_in_and_out_and_a_second_part(self) -> None:
        self.settings.record.duration = 1.0
        ScreenRecorderRecordSettings.start.fire(self.settings.record)
        whole = self.save_held().frames
        self.settings.take.in_out = [0.25, 0.75]
        first = self.save_held().frames
        self.assertEqual(self.span(first), 0.5)
        self.assertEqual(self.times[first[0]] - self.times[whole[0]], 0.25)
        self.settings.take.in_out = [0.75, 1.0]
        second = self.save_held().frames
        self.assertEqual(second[0], first[-1])
        self.assertEqual(second[-1], whole[-1])
        self.assertEqual(len({clip.path for clip in FakeClip.clips}), 3)
        self.discard(ScreenRecorderState.IDLE)

    def test_the_preview_is_the_frame_under_the_dragged_marker(self) -> None:
        self.settings.record.duration = 1.0
        ScreenRecorderRecordSettings.start.fire(self.settings.record)
        whole = self.save_held().frames
        self.assertIsNone(self.recorder.preview_frame)
        self.settings.take.scrub = 0.5
        shown = self.recorder.preview_frame
        assert shown is not None
        self.assertEqual(self.times[tick_of(shown[1])] - self.times[whole[0]], 0.5)
        self.settings.take.scrub = 1.0
        last = self.recorder.preview_frame
        assert last is not None
        self.assertEqual(tick_of(last[1]), whole[-1])
        self.settings.take.scrub = -1.0
        self.assertIsNone(self.recorder.preview_frame)
        self.discard(ScreenRecorderState.IDLE)

    def test_the_playhead_follows_the_dragged_marker(self) -> None:
        self.settings.record.duration = 1.0
        ScreenRecorderRecordSettings.start.fire(self.settings.record)
        self.tick_until(lambda: self.settings.record.state is ScreenRecorderState.HELD)
        self.tick_until(lambda: self.settings.take.playhead[1] == 1.0)
        self.settings.take.scrub = 0.5
        self.tick_until(lambda: self.settings.take.playhead[0] == 0.5)
        self.settings.take.scrub = -1.0
        self.ticks(20)
        self.assertEqual(self.settings.take.playhead[0], 0.5)
        self.discard(ScreenRecorderState.IDLE)

    def test_duration_stops_the_recording(self) -> None:
        self.settings.record.duration = 0.5
        ScreenRecorderRecordSettings.start.fire(self.settings.record)
        frames = self.save_held().frames
        self.assertEqual(self.span(frames), 0.5)
        self.assertEqual(len(frames), frames_for(0.5, FPS, 1) + 1)

    def test_a_take_slowing_down_still_spans_its_duration(self) -> None:
        self.settings.record.duration = 0.5
        ScreenRecorderRecordSettings.start.fire(self.settings.record)
        self.tick_until(lambda: self.stored() >= 4)
        self.interval = 2.0 / FPS
        frames = self.save_held().frames
        self.assertGreaterEqual(self.span(frames), 0.5)
        self.assertLess(self.span(frames), 0.5 + self.interval)
        self.assertLess(len(frames), frames_for(0.5, FPS, 1))

    def test_frame_step_two_records_every_other_tick(self) -> None:
        self.settings.quality.frame_step = 2
        self.settings.record.duration = 0.5
        ScreenRecorderRecordSettings.start.fire(self.settings.record)
        frames = self.save_held().frames
        self.assertEqual(self.span(frames), 0.5)
        self.assertEqual(len(frames), frames_for(0.5, FPS, 2) + 1)
        self.assertEqual(set(np.diff(frames).tolist()), {2})
        self.assertAlmostEqual(FakeClip.clips[0].fps, FPS / 2, places=3)

    def test_the_ram_reserve_stops_the_recording(self) -> None:
        self.free = int(self.settings.record.ram_reserve * GB) + 150 * FRAME_BYTES     # room for two one-second blocks
        ScreenRecorderRecordSettings.start.fire(self.settings.record)
        clip = self.save_held()
        blocks = self.recorder._store.num_slots
        self.assertEqual(len(clip.frames), blocks)      # every slot of the two blocks
        self.assertGreater(blocks, 100)

    def test_max_duration_counts_the_free_ram(self) -> None:
        self.free = int(self.settings.record.ram_reserve * GB) + 640 * FRAME_BYTES     # 10 s at 64 ticks/s
        self.settings.frame.width, self.settings.frame.height = WIDTH, HEIGHT
        self.tick_until(lambda: 0.0 < self.settings.quality.max_duration < 100.0)
        self.assertTrue(640 / 64 - 0.1 <= self.settings.quality.max_duration <= 640 / 60 + 0.1, self.settings.quality.max_duration)
        self.settings.quality.chroma = Chroma.YUV420        # half the bytes a frame: twice the seconds
        self.tick_until(lambda: self.settings.quality.max_duration > 15.0)
        self.assertTrue(1280 / 64 - 0.2 <= self.settings.quality.max_duration <= 1280 / 60 + 0.2, self.settings.quality.max_duration)
        self.settings.quality.bit_depth = BitDepth.BIT_10   # two bytes a sample: half again
        self.tick_until(lambda: self.settings.quality.max_duration < 15.0)
        self.assertTrue(640 / 64 - 0.1 <= self.settings.quality.max_duration <= 640 / 60 + 0.1, self.settings.quality.max_duration)

    def test_the_format_is_read_at_start_and_goes_to_the_clip(self) -> None:
        self.settings.quality.chroma = Chroma.YUV420
        self.settings.quality.bit_depth = BitDepth.BIT_10
        self.settings.record.duration = 0.25
        ScreenRecorderRecordSettings.start.fire(self.settings.record)
        self.tick()
        wanted = ScreenFormat(WIDTH, HEIGHT, Chroma.YUV420, BitDepth.BIT_10)
        self.assertEqual(self.recorder.capture_format, wanted)
        self.settings.quality.chroma = Chroma.YUV444          # a change counts from the next Start
        self.settings.quality.bit_depth = BitDepth.BIT_8
        clip = self.save_held()
        self.assertEqual(clip.format, wanted)
        self.assertEqual(self.recorder._store.shape, wanted.shape)
        self.assertEqual(self.recorder._store.frame(0).dtype, np.uint16)
        self.discard(ScreenRecorderState.IDLE)

    def shown_tick(self) -> int:
        frame = self.recorder.preview_frame
        self.assertIsNotNone(frame)
        assert frame is not None
        return tick_of(frame[1])

    def hold_half_a_second(self) -> list[int]:
        """A held take of 0.5 s; the ticks of its frames, oldest first."""
        self.settings.record.duration = 0.5
        ScreenRecorderRecordSettings.start.fire(self.settings.record)
        self.tick_until(lambda: self.settings.record.state is ScreenRecorderState.HELD)
        return [tick_of(self.recorder._store.frame(slot)) for slot in self.recorder._held_slots]

    def test_play_shows_the_held_frames_at_their_times_looping_between_in_and_out(self) -> None:
        held = self.hold_half_a_second()
        self.settings.take.in_out = [0.5, 1.0]
        self.settings.take.play = True
        shown: list[int] = []
        for _ in range(2 * len(held)):
            self.tick()
            self.assertTrue(self.recorder.playing)
            shown.append(self.shown_tick())
        second_half = held[len(held) // 2:]
        self.assertEqual(shown[0], second_half[0])                       # play starts at in
        self.assertTrue(set(shown) <= set(second_half), shown)
        period = round(0.25 * FPS)          # in to out: half of the 0.5 s, one frame a tick
        self.assertEqual(shown[:period], list(range(second_half[0], second_half[0] + period)), shown)
        self.assertEqual(shown, [shown[n % period] for n in range(len(shown))], shown)    # and loops

    def test_play_goes_off_with_a_save_and_without_a_held_take(self) -> None:
        self.settings.take.play = True
        self.tick()
        self.assertFalse(self.settings.take.play)
        self.assertFalse(self.recorder.playing)
        self.hold_half_a_second()
        self.settings.take.play = True
        self.tick()
        self.assertTrue(self.recorder.playing)
        ScreenRecorderTakeSettings.save.fire(self.settings.take)
        self.tick()
        self.assertFalse(self.settings.take.play)
        self.assertFalse(self.recorder.playing)
        self.tick_until(lambda: self.settings.record.state is ScreenRecorderState.HELD)
        self.assertFalse(self.recorder.playing)
        self.assertIsNone(self.recorder.preview_frame)

    def test_look_back_keeps_the_last_seconds_and_stop_holds_them(self) -> None:
        self.settings.record.duration = 0.5
        self.settings.record.look_back = True
        self.tick()
        self.assertIs(self.settings.record.state, ScreenRecorderState.RECORDING)
        self.tick_until(lambda: self.tick_count > 200)
        ScreenRecorderTakeSettings.save.fire(self.settings.take)         # save only writes a held take
        self.tick()
        self.assertIs(self.settings.record.state, ScreenRecorderState.RECORDING)
        ScreenRecorderRecordSettings.stop.fire(self.settings.record)
        self.tick()
        held_at = self.tick_count
        self.assertIs(self.settings.record.state, ScreenRecorderState.HELD)
        self.ticks(10)
        self.assertFalse(self.recorder.capturing)          # look back pauses while the take is held
        frames = self.save_held().frames
        self.assertGreater(self.recorder._store.num_slots, len(frames))
        self.assertEqual(self.span(frames), 0.5)
        self.assertEqual(np.diff(frames).tolist(), [1] * (len(frames) - 1))
        self.assertLessEqual(frames[-1], held_at)
        slots = self.recorder._store.num_slots
        self.discard(ScreenRecorderState.RECORDING)         # emptied, and recording on in the same store
        self.assertEqual(self.recorder._store.num_slots, slots)
        self.tick_until(lambda: self.stored() >= 5)
        ScreenRecorderRecordSettings.stop.fire(self.settings.record)
        self.tick()
        again = self.save_held().frames
        self.assertGreater(again[0], held_at)               # nothing from before the discard
        self.discard(ScreenRecorderState.RECORDING)

    def test_look_back_off_frees_the_ring(self) -> None:
        self.settings.record.look_back = True
        self.tick_until(lambda: self.recorder._store.num_slots > 0)
        self.settings.record.look_back = False
        self.tick_until(lambda: self.settings.record.state is ScreenRecorderState.IDLE)
        self.assertEqual(self.recorder._store.num_slots, 0)
        self.assertEqual(FakeClip.clips, [])

    def test_a_size_change_remakes_the_look_back_ring(self) -> None:
        self.settings.record.look_back = True
        self.tick_until(lambda: self.recorder._store.num_slots > 0)
        self.settings.frame.width = 16
        self.tick_until(lambda: self.recorder.capture_format == ScreenFormat(16, HEIGHT) and self.stored() > 0)

    def test_a_failure_is_an_error_and_start_tries_again(self) -> None:
        ScreenRecorderRecordSettings.start.fire(self.settings.record)
        self.tick()
        with self.assertLogs('modules.video.screen.screen_recorder', 'ERROR'):
            self.recorder.fail('no capture')
        self.assertIs(self.settings.record.state, ScreenRecorderState.ERROR)
        self.assertIsNone(self.recorder.capture_format)
        self.ticks(3)
        self.assertIs(self.settings.record.state, ScreenRecorderState.ERROR)
        ScreenRecorderRecordSettings.start.fire(self.settings.record)
        self.tick_until(lambda: self.settings.record.state is ScreenRecorderState.RECORDING)

    def test_exit_while_recording_writes_what_was_stored(self) -> None:
        ScreenRecorderRecordSettings.start.fire(self.settings.record)
        self.tick_until(lambda: self.stored() >= 10)
        self.recorder.stop()
        self.assertEqual(len(FakeClip.clips), 1)
        self.assertGreaterEqual(len(FakeClip.clips[0].frames), 10)
        self.assertEqual(self.recorder._store.num_slots, 0)
        self.assertEqual(self.lent, 0)

    def test_exit_while_held_writes_a_take_never_saved(self) -> None:
        self.settings.record.duration = 0.5
        ScreenRecorderRecordSettings.start.fire(self.settings.record)
        self.tick_until(lambda: self.settings.record.state is ScreenRecorderState.HELD)
        self.recorder.stop()
        self.assertEqual(len(FakeClip.clips), 1)

    def test_exit_while_held_after_a_save_writes_nothing_more(self) -> None:
        self.settings.record.duration = 0.5
        ScreenRecorderRecordSettings.start.fire(self.settings.record)
        self.save_held()
        self.recorder.stop()
        self.assertEqual(len(FakeClip.clips), 1)


class QueueTest(unittest.TestCase):
    """The hand-over's queue, its thread not started: a full queue drops its oldest frame and gives its buffer back."""

    def test_a_full_queue_drops_its_oldest_and_counts_it(self) -> None:
        settings = ScreenRecorderSettings()
        settings.frame.width, settings.frame.height = WIDTH, HEIGHT
        recorder = ScreenRecorder(settings, 'unused', FakeClip, lambda: 1 << 40)
        ScreenRecorderRecordSettings.start.fire(settings.record)
        recorder.update(1.0)
        given_back: list[int] = []
        for n in range(6):
            data = np.zeros(ScreenFormat(WIDTH, HEIGHT).shape, np.uint8)
            recorder.submit_screen_frame(ScreenFrame(data, 1.0 + n / FPS, 0, lambda n=n: given_back.append(n)))
        self.assertEqual(given_back, [0, 1, 2])        # the three oldest, dropped
        self.assertEqual(len(recorder._queue), 3)
        self.assertEqual(recorder._dropped, 3)
        recorder.stop()
        self.assertEqual(sorted(given_back), list(range(6)))    # the rest given back at the stop

    def test_a_frame_no_take_wants_is_given_back_at_once(self) -> None:
        recorder = ScreenRecorder(ScreenRecorderSettings(), 'unused', FakeClip, lambda: 1 << 40)
        given_back: list[int] = []
        recorder.submit_screen_frame(ScreenFrame(np.zeros((6, 8), np.uint8), 0.0, 0, lambda: given_back.append(1)))
        self.assertEqual(given_back, [1])
        recorder.stop()


if __name__ == '__main__':
    unittest.main()
