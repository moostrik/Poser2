"""Tests for the grin player: the buffer between the generator and the render (fake clock, no GL)."""

import unittest
from unittest import mock

import numpy as np

from apps.smile.render.GrinPlayer import MARGIN, MIN_FRAMES, GrinPlayer
from apps.smile.render.settings import SequencerSettings

TICK: float = 1 / 60    # the render's frame time


def frame(index: int) -> np.ndarray:
    return np.full((1, 1, 3), index, np.uint8)


def index_of(image: np.ndarray) -> int:
    return int(image[0, 0, 0])


class TestGrinPlayer(unittest.TestCase):
    def setUp(self):
        self.settings = SequencerSettings()
        self.settings.grin_fps = 15.0
        self.settings.grin_delay = 1.0
        self.grin_player = GrinPlayer(self.settings)
        self.now = 0.0
        patcher = mock.patch('apps.smile.render.GrinPlayer.time.perf_counter', lambda: self.now)
        patcher.start()
        self.addCleanup(patcher.stop)

    def play(self, total: int, made_fps: float, hold: bool, first: float = 0.5, portrait: float | None = None) -> list[tuple[float, int]]:
        """One clip: its frames arrive at *made_fps* from *first* on, the render asks every tick.
        Returns (time, index) of every frame shown."""
        arrivals: list[float] = [first + i / made_fps for i in range(total)]
        shown: list[tuple[float, int]] = []
        added: int = 0
        tick: int = 0
        self.grin_player.begin(total, hold)
        while self.grin_player.busy:
            now: float = tick * TICK
            if portrait is not None and portrait <= now:
                self.now = portrait
                self.grin_player.hold_from_now()
                portrait = None
            while added < total and arrivals[added] <= now:
                self.now = arrivals[added]
                self.grin_player.add(frame(added))
                added += 1
                if added == total:
                    self.grin_player.end()
            self.now = now
            image = self.grin_player.update(now)
            if image is not None:
                shown.append((now, index_of(image)))
            tick += 1
            self.assertLess(now, 120.0, 'the clip never finished')
        return shown

    def assert_steady(self, shown: list[tuple[float, int]]) -> None:
        """From the second frame on (the first is the hold): every frame, in order, one interval apart."""
        self.assertEqual([index for _, index in shown], list(range(len(shown))))
        interval: float = 1 / self.settings.grin_fps
        for (a, _), (b, _) in zip(shown[1:], shown[2:]):
            self.assertAlmostEqual(b - a, interval, delta=TICK + 1e-6)

    def test_first_frame_at_once_then_held_for_the_delay(self):
        shown = self.play(total=30, made_fps=100.0, hold=True, first=0.5)
        self.assertAlmostEqual(shown[0][0], 0.5, delta=TICK)
        self.assertAlmostEqual(shown[1][0], 0.5 + self.settings.grin_delay, delta=TICK + 1e-6)
        self.assert_steady(shown)

    def test_the_hold_counts_from_the_portrait(self):
        shown = self.play(total=30, made_fps=100.0, hold=True, first=0.5, portrait=0.2)
        self.assertAlmostEqual(shown[1][0], 0.2 + self.settings.grin_delay, delta=TICK + 1e-6)

    def test_every_frame_once_in_order_at_the_playback_speed(self):
        shown = self.play(total=60, made_fps=100.0, hold=False)
        self.assertEqual(len(shown), 60)
        self.assert_steady(shown)
        self.assertEqual(self.grin_player.plays, 1)
        self.assertEqual(self.grin_player.progress, 1.0)

    def test_frames_made_slower_than_played_start_later_and_never_run_dry(self):
        total, made_fps, first = 100, 10.0, 0.5
        shown = self.play(total, made_fps, hold=False, first=first)
        self.assertEqual(len(shown), total)
        self.assert_steady(shown)
        # the pause is in front: it ends just after ComfyUI's last frame, not later
        last_arrival: float = first + (total - 1) / made_fps
        self.assertGreaterEqual(shown[-1][0], last_arrival)
        self.assertLess(shown[-1][0], last_arrival + MARGIN + 0.5)

    def test_frames_made_faster_than_played_start_when_the_speed_is_known(self):
        first, made_fps = 0.5, 30.0
        shown = self.play(total=60, made_fps=made_fps, hold=False, first=first)
        self.assertAlmostEqual(shown[1][0], first + (MIN_FRAMES - 1) / made_fps, delta=2 * TICK)
        self.assert_steady(shown)

    def test_a_slow_delay_setting_is_not_added_to_the_buffer_wait(self):
        # 100 frames at 10 per second need about 3.5 s of buffer: a delay of 1 s changes nothing
        held = self.play(total=100, made_fps=10.0, hold=True)
        start_held: float = held[1][0]
        self.grin_player.clear()
        replay = self.play(total=100, made_fps=10.0, hold=False)
        self.assertAlmostEqual(start_held, replay[1][0], delta=2 * TICK)

    def test_one_frame_clip_shows_and_counts_as_played(self):
        shown = self.play(total=1, made_fps=10.0, hold=False)
        self.assertEqual([index for _, index in shown], [0])
        self.assertEqual(self.grin_player.plays, 1)
        self.assertTrue(self.grin_player.playing)

    def test_busy_until_the_last_frame_is_shown(self):
        self.grin_player.begin(3, hold=False)
        for i in range(3):
            self.grin_player.add(frame(i))
        self.grin_player.end()
        self.assertTrue(self.grin_player.busy)
        self.assertEqual(index_of(self.grin_player.update(0.0)), 0)
        self.assertIsNone(self.grin_player.update(0.01))
        self.assertEqual(index_of(self.grin_player.update(1 / 15)), 1)
        self.assertTrue(self.grin_player.busy)
        self.assertEqual(index_of(self.grin_player.update(2 / 15)), 2)
        self.assertFalse(self.grin_player.busy)

    def test_a_clip_without_frames_is_not_busy(self):
        self.grin_player.begin(30, hold=True)
        self.grin_player.end()
        self.assertFalse(self.grin_player.busy)
        self.assertIsNone(self.grin_player.update(1.0))

    def test_clear_drops_the_clip_and_late_frames(self):
        self.grin_player.begin(30, hold=False)
        self.grin_player.add(frame(0))
        self.grin_player.clear()
        self.assertFalse(self.grin_player.busy)
        self.grin_player.add(frame(1))       # arrives after the reset
        self.assertIsNone(self.grin_player.update(1.0))
        self.assertEqual(self.grin_player.plays, 0)

    def test_frames_without_a_clip_are_dropped(self):
        self.grin_player.add(frame(0))       # the warm-up opens no clip
        self.grin_player.end()
        self.assertFalse(self.grin_player.busy)
        self.assertIsNone(self.grin_player.update(1.0))

    def test_running_dry_continues_without_a_burst(self):
        self.grin_player.begin(12, hold=False)
        for i in range(10):
            self.now = i / 20
            self.grin_player.add(frame(i))
        shown: list[int] = []
        for tick in range(120):         # two seconds: the ten frames play, then it waits for the last two
            image = self.grin_player.update(tick * TICK)
            if image is not None:
                shown.append(index_of(image))
        self.assertEqual(shown, list(range(10)))
        self.now = 2.0
        self.grin_player.add(frame(10))
        self.grin_player.add(frame(11))
        self.grin_player.end()
        self.assertEqual(index_of(self.grin_player.update(2.0)), 10)
        self.assertIsNone(self.grin_player.update(2.0 + TICK))
        self.assertEqual(index_of(self.grin_player.update(2.0 + 1 / 15)), 11)

    def test_reports_the_speed_and_the_waits(self):
        self.play(total=100, made_fps=10.0, hold=True, first=0.5, portrait=0.2)
        self.assertAlmostEqual(self.settings.grin_made_fps, 10.0, delta=0.1)
        # the last frame arrives at 0.5 + 9.9 s and has its turn 98 / 15 s after the start; the waits count from the portrait
        needed: float = 0.5 + 99 / 10.0 + MARGIN - 98 / 15.0 - 0.2
        self.assertAlmostEqual(self.settings.grin_buffer_wait, needed, delta=0.05)
        self.assertAlmostEqual(self.settings.grin_start_wait, needed, delta=0.1)


if __name__ == '__main__':
    unittest.main()
