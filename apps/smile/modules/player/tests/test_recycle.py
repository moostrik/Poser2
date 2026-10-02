"""Tests for recycling: a clip from disk joins whenever no clip has joined for a while."""

import tempfile
import unittest

import numpy as np

from modules.player.ClipPool import Clip, ClipPool
from modules.player.FolderScanner import FolderScanner
from modules.player.tests.test_even_spread import turns
from modules.player.tests.test_follow_style import FakePool, touch


class Clock:
    def __init__(self) -> None:
        self.now: float = 0.0

    def __call__(self) -> float:
        return self.now


def started(folder: str, pool: FakePool, max_files: int = 4, recent: int = 2, every: float = 20.0) -> tuple[FolderScanner, Clock]:
    """A scanner that has done its startup listing at time 0."""
    scanner = FolderScanner(folder, pool, max_files, recent, every)
    clock = Clock()
    scanner._clock = clock
    scanner.update()
    return scanner, clock


def recycled(pool: FakePool, since: int) -> list[str]:
    return [id for id, _ in pool.added[since:]]


class TestRecycling(unittest.TestCase):

    def test_nothing_before_the_wait_has_passed_then_one_clip(self):
        with tempfile.TemporaryDirectory() as folder:
            for i in range(10):
                touch(folder, 'A', f'{i:02d}')
            pool = FakePool(capacity=4)
            scanner, clock = started(folder, pool)
            loaded = set(pool.loaded)
            self.assertEqual(len(pool.added), 4)
            clock.now = 19.0
            scanner.tick()
            self.assertEqual(len(pool.added), 4)
            clock.now = 20.0
            scanner.tick()
            self.assertEqual(len(pool.added), 5)
            id, style = pool.added[-1]
            self.assertEqual(style, 'A')
            self.assertNotIn(id, loaded)                # one that was not in the pool
            self.assertEqual(pool.archive[-1], id)      # it joins the archive, not the recent list

    def test_one_clip_per_wait(self):
        with tempfile.TemporaryDirectory() as folder:
            for i in range(10):
                touch(folder, 'A', f'{i:02d}')
            pool = FakePool(capacity=4)
            scanner, clock = started(folder, pool)
            for second in range(1, 101):                # the thread ticks once a second
                clock.now = float(second)
                scanner.tick()
            self.assertEqual(len(pool.added) - 4, 5)    # at 20, 40, 60, 80 and 100 s

    def test_a_recording_postpones_the_next_clip_from_disk(self):
        with tempfile.TemporaryDirectory() as folder:
            for i in range(10):
                touch(folder, 'A', f'{i:02d}')
            pool = FakePool(capacity=4)
            scanner, clock = started(folder, pool)
            clock.now = 15.0
            touch(folder, 'A', '99')
            scanner.update()
            self.assertEqual(pool.recent[-1], '99')
            clock.now = 34.0
            scanner.tick()
            self.assertEqual(len(pool.added), 5)        # the recording only
            clock.now = 35.0
            scanner.tick()
            self.assertEqual(len(pool.added), 6)

    def test_nothing_while_the_pool_is_still_decoding(self):
        with tempfile.TemporaryDirectory() as folder:
            for i in range(10):
                touch(folder, 'A', f'{i:02d}')
            pool = FakePool(capacity=4)
            scanner, clock = started(folder, pool)
            pool.idle = False
            clock.now = 60.0
            scanner.tick()
            self.assertEqual(len(pool.added), 4)
            pool.idle = True
            scanner.tick()
            self.assertEqual(len(pool.added), 5)

    def test_nothing_when_every_clip_on_disk_is_in_the_pool(self):
        with tempfile.TemporaryDirectory() as folder:
            for i in range(3):
                touch(folder, 'A', f'{i:02d}')
            pool = FakePool(capacity=4)
            scanner, clock = started(folder, pool)
            for second in range(0, 600, 5):
                clock.now = float(second)
                scanner.tick()
            self.assertEqual(len(pool.added), 3)        # no clip is loaded twice

    def test_only_clips_of_the_current_style(self):
        with tempfile.TemporaryDirectory() as folder:
            for i in range(8):
                touch(folder, 'B', f'{i:02d}')
            for i in range(10, 18):
                touch(folder, 'A', f'{i:02d}')
            pool = FakePool(capacity=4)
            scanner, clock = started(folder, pool)
            self.assertEqual(pool.style, 'A')
            for step in range(1, 13):
                clock.now = step * 20.0
                scanner.tick()
            self.assertEqual(len(pool.added), 16)
            self.assertTrue(all(style == 'A' for _, style in pool.added))

    def test_every_clip_comes_round_before_any_repeats(self):
        with tempfile.TemporaryDirectory() as folder:
            for i in range(10):
                touch(folder, 'A', f'{i:02d}')
            pool = FakePool(capacity=4)
            scanner, clock = started(folder, pool)
            waiting = {f'{i:02d}' for i in range(10)} - set(pool.loaded)
            for step in range(1, 7):
                clock.now = step * 20.0
                scanner.tick()
            self.assertEqual(set(recycled(pool, 4)), waiting)       # the six that were outside, each once

    def test_a_few_more_clips_than_the_pool_take_turns(self):
        with tempfile.TemporaryDirectory() as folder:
            for i in range(7):
                touch(folder, 'A', f'{i:02d}')
            pool = FakePool(capacity=4)
            scanner, clock = started(folder, pool)
            for step in range(1, 34):
                clock.now = step * 20.0
                scanner.tick()
            rounds = recycled(pool, 4)
            self.assertEqual(len(rounds), 33)
            for start in range(0, 33, 3):               # three are outside the pool at any time
                self.assertEqual(len(set(rounds[start:start + 3])), 3)

    def test_no_recycling_when_switched_off(self):
        with tempfile.TemporaryDirectory() as folder:
            for i in range(10):
                touch(folder, 'A', f'{i:02d}')
            pool = FakePool(capacity=4)
            scanner, clock = started(folder, pool, every=0.0)
            clock.now = 1000.0
            scanner.tick()
            self.assertEqual(len(pool.added), 4)

    def test_nothing_before_the_first_clip(self):
        with tempfile.TemporaryDirectory() as folder:
            pool = FakePool()
            scanner, clock = started(folder, pool)
            clock.now = 100.0
            scanner.tick()
            self.assertEqual(pool.added, [])

    def test_after_a_switch_the_clips_come_from_the_new_style(self):
        with tempfile.TemporaryDirectory() as folder:
            for i in range(8):
                touch(folder, 'B', f'{i:02d}')
            for i in range(10, 18):
                touch(folder, 'A', f'{i:02d}')
            pool = FakePool(capacity=4)
            scanner, clock = started(folder, pool)
            clock.now = 20.0
            scanner.tick()                              # one A clip, and a deck of A clips
            clock.now = 30.0
            touch(folder, 'B', '50')
            scanner.update()
            self.assertEqual(pool.style, 'B')
            count = len(pool.added)
            clock.now = 49.0
            scanner.tick()
            self.assertEqual(len(pool.added), count)    # the switch counts as clips joining
            for step in range(5):
                clock.now = 50.0 + step * 20.0
                scanner.tick()
            self.assertEqual(len(pool.added), count + 5)        # at 50, 70, 90, 110 and 130 s
            self.assertTrue(all(style == 'B' for _, style in pool.added[count:]))


class TestTheWallWhileRecycling(unittest.TestCase):

    def test_the_wall_stays_even_and_each_recycled_clip_is_shown(self):
        pool = ClipPool(30, (2, 2), recent=4)
        for i in range(30):
            pool._insert(Clip(f'{i:03d}', np.zeros((1, 2, 2, 3), np.uint8), 15.0, 'A'), recent=i >= 26)
        last = list(pool._recent)
        playing: list[Clip | None] = [None] * 24
        waiting: dict[str, tuple[Clip, int]] = {}       # recycled clips not shown yet, and the step they joined
        for step, cell in enumerate(turns(24, 100)):
            playing[cell] = pool.get_clip(playing[cell])
            waiting.pop(playing[cell].id, None)
            if step > 100:
                self.assertLessEqual(pool.spread()[1], 1)                           # no clip on two cells
                self.assertLessEqual(sum(1 for c in last if c.active == 0), 1)      # the last recordings stay
                for recycled_clip, joined in waiting.values():
                    self.assertLess(step - joined, 32, recycled_clip.id)           # shown within about a round
                if step % 12 == 0:                                                  # a clip from disk joins
                    new = Clip(f'd{step:05d}', np.zeros((1, 2, 2, 3), np.uint8), 15.0, 'A')
                    pool._insert(new)
                    waiting[new.id] = (new, step)
        self.assertEqual(len(pool), 30)
        self.assertEqual([c.id for c in pool._recent], [c.id for c in last])
