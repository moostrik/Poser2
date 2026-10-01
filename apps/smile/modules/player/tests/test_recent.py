"""Tests for the newest clips that are always on the wall."""

import os
import random
import tempfile
import unittest

import numpy as np

from modules.player.ClipPool import Clip, ClipPool, choose
from modules.player.FolderScanner import FolderScanner
from modules.player.tests.test_even_spread import turns
from modules.player.tests.test_follow_style import FakePool, touch


def clip(id: str, plays: int = 0, active: int = 0, fresh: bool = False) -> Clip:
    return Clip(id, np.zeros((1, 2, 2, 3), np.uint8), 15.0, 'A', plays, active, fresh)


def pool_of(count: int, size: int | None = None, recent: int = 3) -> ClipPool:
    pool = ClipPool(size or count, (2, 2), recent=recent)
    for i in range(count):
        pool._insert(Clip(f'{i:03d}', np.zeros((1, 2, 2, 3), np.uint8), 15.0, 'A'))
    return pool


class TestChooseRecent(unittest.TestCase):

    def test_a_recent_clip_on_no_cell_goes_before_an_older_unplayed_one(self):
        clips = [clip('01', plays=0, fresh=True), clip('02', plays=5, active=1), clip('08', plays=9), clip('09', plays=9, active=1), clip('10', plays=9, active=1)]
        self.assertEqual({choose(clips, recent=3).id for _ in range(30)}, {'08'})

    def test_recent_clips_that_are_all_shown_leave_the_rule_as_it_is(self):
        clips = [clip('01', plays=4), clip('02', plays=2), clip('08', active=1), clip('09', active=1), clip('10', active=1)]
        self.assertEqual({choose(clips, recent=3).id for _ in range(30)}, {'02'})

    def test_the_cell_that_played_a_recent_clip_does_not_take_it_back(self):
        clips = [clip('01', plays=4), clip('02', plays=2), clip('08', active=1), clip('09', active=1), clip('10', active=1)]
        just_played = clips[4]
        self.assertEqual({choose(clips, exclude=just_played, recent=3).id for _ in range(30)}, {'02'})

    def test_the_next_cell_picks_up_a_recent_clip_that_ended(self):
        clips = [clip('01', plays=4), clip('02', plays=2), clip('08', active=1), clip('09', active=1), clip('10', plays=30)]
        self.assertEqual({choose(clips, exclude=clips[0], recent=3).id for _ in range(30)}, {'10'})

    def test_recent_zero_is_the_plain_rule(self):
        clips = [clip('01', plays=0), clip('10', plays=9)]
        self.assertEqual({choose(clips, recent=0).id for _ in range(30)}, {'01'})


class TestRecentOnTheWall(unittest.TestCase):

    def test_the_three_newest_are_always_shown(self):
        pool = pool_of(30)
        newest = sorted(pool.clips(), key=lambda c: c.id)[-3:]
        playing: list[Clip | None] = [None] * 24
        missing_before: set[str] = set()
        seen: set[str] = set()
        for step, cell in enumerate(turns(24, 120)):
            playing[cell] = pool.get_clip(playing[cell])
            seen.add(playing[cell].id)
            if step > 100:
                self.assertEqual(pool.spread(), (24, 1))                # the others still spread evenly
                missing = {c.id for c in newest if c.active == 0}
                self.assertLessEqual(len(missing), 1)                   # only the one that just ended
                self.assertFalse(missing & missing_before)              # and it is back at the next step
                missing_before = missing
        self.assertEqual(len(seen), 30)
        others = [c.plays for c in pool.clips() if c not in newest]
        self.assertLessEqual(max(others) - min(others), 3)

    def test_a_new_clip_joins_the_three_and_the_oldest_of_them_leaves(self):
        pool = pool_of(30, size=31)
        was_third = sorted(pool.clips(), key=lambda c: c.id)[-3]
        playing: list[Clip | None] = [None] * 24
        for cell in turns(24, 40):
            playing[cell] = pool.get_clip(playing[cell])
        new = Clip('999', np.zeros((1, 2, 2, 3), np.uint8), 15.0, 'A')
        pool._insert(new)
        cell = random.randrange(24)
        playing[cell] = pool.get_clip(playing[cell])
        self.assertIs(playing[cell], new)

        off_the_wall = 0
        for step, cell in enumerate(turns(24, 60)):
            playing[cell] = pool.get_clip(playing[cell])
            if step > 24:
                self.assertLessEqual(sum(1 for c in playing if c is new), 1)
                off_the_wall += was_third.active == 0
        self.assertEqual(new.active, 1)
        self.assertGreater(off_the_wall, 24)        # back in the normal rotation: off the wall now and then


class TestScannerLoadsTheNewest(unittest.TestCase):

    def test_a_switch_loads_the_newest_and_fills_up_at_random(self):
        with tempfile.TemporaryDirectory() as folder:
            for i in range(1, 21):
                touch(folder, 'B', f'{i:02d}')
            touch(folder, 'A', '30')
            pool = FakePool()
            scanner = FolderScanner(folder, pool, 6, follow_style=True, recent=3)
            scanner.update()
            pool.added.clear()

            touch(folder, 'B', '40')
            scanner.update()
            ids = [id for id, _ in pool.added]
            self.assertEqual(len(ids), 6)
            self.assertEqual(ids[-3:], ['19', '20', '40'])      # queued last: decoded first, the newest first of all
            self.assertEqual(len(set(ids)), 6)
            self.assertTrue(set(ids[:3]) <= {f'{i:02d}' for i in range(1, 19)})

    def test_startup_loads_the_three_newest_of_the_style(self):
        with tempfile.TemporaryDirectory() as folder:
            for i in range(1, 11):
                touch(folder, 'A', f'{i:02d}')
            touch(folder, 'B', '05b')
            pool = FakePool()
            scanner = FolderScanner(folder, pool, 5, follow_style=True, recent=3)
            scanner.update()
            ids = [id for id, _ in pool.added]
            self.assertEqual(ids[-3:], ['08', '09', '10'])
            self.assertEqual(len(ids), 5)

    def test_fewer_clips_than_recent(self):
        with tempfile.TemporaryDirectory() as folder:
            touch(folder, 'A', '01')
            touch(folder, 'A', '02')
            pool = FakePool()
            scanner = FolderScanner(folder, pool, 5, follow_style=True, recent=3)
            scanner.update()
            self.assertEqual([id for id, _ in pool.added], ['01', '02'])
