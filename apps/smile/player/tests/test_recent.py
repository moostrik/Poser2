"""Tests for the last recordings, which are always on the wall."""

import random
import tempfile
import unittest

import numpy as np

from apps.smile import sphere
from apps.smile.player.ClipPool import Clip, ClipPool, choose
from apps.smile.player.FolderScanner import FolderScanner
from apps.smile.player.tests.test_even_spread import turns
from apps.smile.player.tests.test_follow_style import FakePool, touch


def clip(id: str, plays: int = 0, active: int = 0) -> Clip:
    return Clip(id, np.zeros((1, 2, 2, 3), np.uint8), 15.0, 'A', plays, active, [active, 0, 0])


def pool_of(count: int, size: int | None = None, recent: int = 3) -> ClipPool:
    """A pool of `count` clips, the newest `recent` of them recordings."""
    pool = ClipPool(size or count, (2, 2), recent=recent)
    for i in range(count):
        pool._insert(clip(f'{i:03d}'), recent=i >= count - recent)
    return pool


class TestChooseRecent(unittest.TestCase):

    def test_a_recent_clip_on_no_cell_goes_before_an_older_unplayed_one(self):
        clips = [clip('01', plays=0), clip('02', plays=5, active=1), clip('08', plays=9), clip('09', plays=9, active=1), clip('10', plays=9, active=1)]
        self.assertEqual({choose(clips, recent=clips[2:]).id for _ in range(30)}, {'08'})

    def test_recent_clips_that_are_all_shown_leave_the_rule_as_it_is(self):
        clips = [clip('01', plays=4), clip('02', plays=2), clip('08', active=1), clip('09', active=1), clip('10', active=1)]
        self.assertEqual({choose(clips, recent=clips[2:]).id for _ in range(30)}, {'02'})

    def test_the_cell_that_played_a_recent_clip_does_not_take_it_back(self):
        clips = [clip('01', plays=4), clip('02', plays=2), clip('08', active=1), clip('09', active=1), clip('10', active=1)]
        just_played = clips[4]
        self.assertEqual({choose(clips, exclude=just_played, recent=clips[2:]).id for _ in range(30)}, {'02'})

    def test_the_next_cell_picks_up_a_recent_clip_that_ended(self):
        clips = [clip('01', plays=4), clip('02', plays=2), clip('08', active=1), clip('09', active=1), clip('10', plays=30)]
        self.assertEqual({choose(clips, exclude=clips[0], recent=clips[2:]).id for _ in range(30)}, {'10'})

    def test_without_recent_clips_it_is_the_plain_rule(self):
        clips = [clip('01', plays=0), clip('10', plays=9)]
        self.assertEqual({choose(clips).id for _ in range(30)}, {'01'})


class TestRecentOnTheWall(unittest.TestCase):

    def test_the_last_recordings_are_always_shown(self):
        pool = pool_of(30, recent=4)
        last = list(pool._recent)
        playing: list[Clip | None] = [None] * 24
        missing_before: set[str] = set()
        seen: set[str] = set()
        for step, cell in enumerate(turns(24, 120)):
            playing[cell] = pool.get_clip(playing[cell])
            seen.add(playing[cell].id)
            if step > 100:
                self.assertEqual(pool.spread(), (24, 1))                # the others still spread evenly
                missing = {c.id for c in last if c.active == 0}
                self.assertLessEqual(len(missing), 1)                   # only the one that just ended
                self.assertFalse(missing & missing_before)              # and it is back at the next step
                missing_before = missing
        self.assertEqual(len(seen), 30)
        others = [c.plays for c in pool._archive]
        self.assertLessEqual(max(others) - min(others), 6)      # about 92 plays each over the run, within a few

    def test_a_new_recording_is_shown_at_once_and_the_oldest_recent_one_takes_turns_again(self):
        pool = pool_of(30, size=31)
        was_third = pool._recent[0]
        playing: list[Clip | None] = [None] * 24
        for cell in turns(24, 40):
            playing[cell] = pool.get_clip(playing[cell])
        new = clip('999')
        pool._insert(new, recent=True)
        self.assertIn(was_third, pool._archive)
        cell = random.randrange(24)
        playing[cell] = pool.get_clip(playing[cell])
        self.assertIs(playing[cell], new)

        off_the_wall = shown = 0
        new_was_missing = False
        for step, cell in enumerate(turns(24, 60)):
            playing[cell] = pool.get_clip(playing[cell])
            if step > 24:
                self.assertLessEqual(sum(1 for c in playing if c is new), 1)
                # the new recording stays on the wall: missing only for the step in which its cell ended
                self.assertFalse(new_was_missing and new.active == 0)
                new_was_missing = new.active == 0
                off_the_wall += was_third.active == 0
                shown += was_third.active == 1
        self.assertGreater(off_the_wall, 24)        # in the normal rotation: off the wall now and then
        self.assertGreater(shown, off_the_wall)     # and on it most of the time, like any archive clip


class TestRecentOnEveryThird(unittest.TestCase):
    """The sphere's 48 cells in three sectors of 16: every recent recording plays on each of them."""

    SECTORS: list[int] = [sphere.sector_of_cell(*divmod(id, sphere.CELLS)) for id in range(sphere.SCREENS)]

    def test_the_cells_fall_into_three_equal_thirds(self):
        self.assertEqual([self.SECTORS.count(s) for s in range(sphere.SECTORS)], [16, 16, 16])

    def test_every_recent_recording_is_on_every_third(self):
        pool = pool_of(30, recent=4)
        last = list(pool._recent)
        playing: list[Clip | None] = [None] * 48
        missing_before: set[tuple[str, int]] = set()
        for step, cell in enumerate(turns(48, 60)):
            sector: int = self.SECTORS[cell]
            playing[cell] = pool.get_clip(playing[cell], sector)
            if step > 150:
                for c in pool.clips():      # the bookkeeping per third matches what the cells play
                    self.assertEqual(c.on, [sum(1 for i, p in enumerate(playing) if p is c and self.SECTORS[i] == s) for s in range(3)])
                missing = {(c.id, s) for c in last for s in range(3) if c.on[s] == 0}
                self.assertLessEqual(len(missing), 3)                   # at most the one whose cell just ended, per third
                # a recording missing from a third is taken by the next cell that finishes there
                self.assertFalse({(id, s) for id, s in missing_before if s == sector} & missing)
                missing_before = missing
        self.assertEqual(pool.spread()[1], 3)                           # a recent clip is on each third once


class TestScannerLoadsTheNewest(unittest.TestCase):

    def test_a_switch_loads_the_newest_as_recent_and_fills_the_archive_at_random(self):
        with tempfile.TemporaryDirectory() as folder:
            for i in range(1, 21):
                touch(folder, 'B', f'{i:02d}')
            touch(folder, 'A', '30')
            pool = FakePool()
            scanner = FolderScanner(folder, pool, 6, recent=3)
            scanner.update()
            pool.added.clear()
            pool.recent.clear()
            pool.archive.clear()

            touch(folder, 'B', '40')
            scanner.update()
            ids = [id for id, _ in pool.added]
            self.assertEqual(len(ids), 6)
            self.assertEqual(ids[-3:], ['19', '20', '40'])      # queued last: decoded first, the newest first of all
            self.assertEqual(pool.recent, ['19', '20', '40'])
            self.assertEqual(len(set(pool.archive)), 3)
            self.assertTrue(set(pool.archive) <= {f'{i:02d}' for i in range(1, 19)})

    def test_startup_loads_the_newest_of_the_style(self):
        with tempfile.TemporaryDirectory() as folder:
            for i in range(1, 11):
                touch(folder, 'A', f'{i:02d}')
            touch(folder, 'B', '05b')
            pool = FakePool()
            scanner = FolderScanner(folder, pool, 5, recent=3)
            scanner.update()
            ids = [id for id, _ in pool.added]
            self.assertEqual(ids[-3:], ['08', '09', '10'])
            self.assertEqual(pool.recent, ['08', '09', '10'])
            self.assertEqual(len(ids), 5)

    def test_fewer_clips_than_recent(self):
        with tempfile.TemporaryDirectory() as folder:
            touch(folder, 'A', '01')
            touch(folder, 'A', '02')
            pool = FakePool()
            scanner = FolderScanner(folder, pool, 5, recent=3)
            scanner.update()
            self.assertEqual([id for id, _ in pool.added], ['01', '02'])
            self.assertEqual(pool.recent, ['01', '02'])
