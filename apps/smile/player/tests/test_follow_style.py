"""Tests for the players following SMILE's style: which clips the scanner hands to the pool and which the cells get."""

import os
import tempfile
import unittest

import numpy as np

from apps.smile.player.ClipPool import Clip, ClipPool
from apps.smile.player.FolderScanner import FolderScanner, clip_style, find_clips, pick_clips


def clip(id: str, style: str) -> Clip:
    return Clip(id, np.zeros((1, 2, 2, 3), np.uint8), 15.0, style)


def touch(folder: str, style: str, id: str) -> str:
    os.makedirs(os.path.join(folder, style), exist_ok=True)
    path = os.path.join(folder, style, id + '.mp4')
    open(path, 'w').close()
    return path


class FakePool:
    """Records what the scanner hands over. `capacity` makes it drop its oldest like the pool, for
    the tests that need to know what is loaded."""

    def __init__(self, capacity: int = 1000) -> None:
        self.added: list[tuple[str, str]] = []      # (id, style) in the order they were queued
        self.recent: list[str] = []                 # the ids handed over as recordings
        self.archive: list[str] = []                # the ids handed over as clips from disk
        self.removed: list[str] = []
        self.loaded: list[str] = []
        self.capacity: int = capacity
        self.style: str | None = None
        self.idle: bool = True

    def add(self, path: str, style: str = '', recent: bool = False) -> None:
        id: str = os.path.splitext(os.path.basename(path))[0]
        self.added.append((id, style))
        (self.recent if recent else self.archive).append(id)
        self.loaded.append(id)
        while len(self.loaded) > self.capacity:
            self.loaded.pop(0)

    def remove(self, id: str) -> None:
        self.removed.append(id)

    def switch(self, style: str | None) -> None:
        self.style = style

    def ids(self) -> set[str]:
        return set(self.loaded)


class TestClipFiles(unittest.TestCase):

    def test_style_is_the_folder(self):
        self.assertEqual(clip_style(os.path.join('output', 'grins', 'B', '2026_10_01_12_00_00.mp4')), 'B')

    def test_find_clips_ignores_files_without_a_style_folder(self):
        with tempfile.TemporaryDirectory() as folder:
            touch(folder, 'A', '01')
            touch(folder, 'B', '02')
            open(os.path.join(folder, 'old_A.mp4'), 'w').close()
            self.assertEqual(sorted(find_clips(folder)), ['01', '02'])

    def test_pick_clips_takes_only_the_style_and_at_most_count(self):
        paths = [os.path.join('g', 'A', f'{i:02d}.mp4') for i in range(10)] + [os.path.join('g', 'B', f'{i:02d}.mp4') for i in range(10, 13)]
        picked = pick_clips(paths, 'A', 4)
        self.assertEqual(len(picked), 4)
        self.assertEqual(len(set(picked)), 4)
        self.assertTrue(all(clip_style(p) == 'A' for p in picked))
        self.assertEqual(sorted(pick_clips(paths, 'B', 24)), sorted(paths[10:]))
        self.assertEqual(pick_clips(paths, 'B', 0), [])


class TestPoolStyle(unittest.TestCase):

    def pool(self, style: str | None, clips: list[Clip]) -> ClipPool:
        pool = ClipPool(4, (2, 2))
        pool._archive = list(clips)
        pool.style = style
        return pool

    def test_get_clip_takes_the_current_style(self):
        pool = self.pool('B', [clip('01', 'A'), clip('02', 'A'), clip('03', 'B')])
        self.assertEqual({pool.get_clip().id for _ in range(50)}, {'03'})

    def test_get_clip_falls_back_while_the_style_has_no_clip(self):
        pool = self.pool('B', [clip('01', 'A'), clip('02', 'A')])
        self.assertIn(pool.get_clip().id, ('01', '02'))

    def test_without_a_style_every_clip_can_be_taken(self):
        pool = self.pool(None, [clip('01', 'A'), clip('02', 'B')])
        self.assertEqual({pool.get_clip().id for _ in range(200)}, {'01', '02'})

    def test_clips_for_the_start_are_of_the_current_style(self):
        pool = self.pool('A', [clip('01', 'A'), clip('02', 'B'), clip('03', 'A')])
        self.assertEqual([c.id for c in pool.clips()], ['01', '03'])

    def test_a_clip_of_the_new_style_pushes_out_the_other_style(self):
        pool = self.pool('A', [clip('01', 'A'), clip('02', 'A'), clip('03', 'A'), clip('04', 'A')])
        pool.switch('B')
        pool._insert(clip('05', 'B'), recent=True)
        self.assertEqual(sorted(pool.ids()), ['02', '03', '04', '05'])
        self.assertEqual({pool.get_clip().id for _ in range(20)}, {'05'})


class TestScannerFollowsStyle(unittest.TestCase):

    def test_startup_takes_the_style_of_the_newest_clip(self):
        with tempfile.TemporaryDirectory() as folder:
            for id in ('01', '02', '03'):
                touch(folder, 'A', id)
            touch(folder, 'B', '04')
            touch(folder, 'B', '05')
            pool = FakePool()
            scanner = FolderScanner(folder, pool, 24)
            scanner.update()
            self.assertEqual(pool.style, 'B')
            self.assertEqual(sorted(pool.added), [('04', 'B'), ('05', 'B')])
            self.assertEqual(pool.added[-1], ('05', 'B'))      # the newest is queued last: decoded first

    def test_a_new_recording_of_the_current_style_joins_recent(self):
        with tempfile.TemporaryDirectory() as folder:
            touch(folder, 'A', '01')
            pool = FakePool()
            scanner = FolderScanner(folder, pool, 24)
            scanner.update()
            touch(folder, 'A', '02')
            scanner.update()
            self.assertEqual(pool.added, [('01', 'A'), ('02', 'A')])
            self.assertEqual(pool.recent, ['01', '02'])
            self.assertEqual(pool.style, 'A')

    def test_a_recording_of_the_other_style_switches(self):
        with tempfile.TemporaryDirectory() as folder:
            for id in ('01', '02', '03', '04'):
                touch(folder, 'B', id)      # an earlier B period
            for id in ('05', '06'):
                touch(folder, 'A', id)
            pool = FakePool()
            scanner = FolderScanner(folder, pool, 3)
            scanner.update()
            self.assertEqual(pool.style, 'A')
            pool.added.clear()
            pool.recent.clear()
            pool.archive.clear()

            touch(folder, 'B', '07')
            scanner.update()
            self.assertEqual(pool.style, 'B')
            self.assertEqual(len(pool.added), 3)                # two older B clips and the new one
            self.assertEqual(pool.added[-1], ('07', 'B'))       # queued last: decoded first
            self.assertTrue(all(style == 'B' for _, style in pool.added))
            self.assertEqual(pool.recent, ['07'])               # the clip that caused the switch is the recording
            self.assertTrue(set(pool.archive) <= {'01', '02', '03', '04'})      # the clips from disk are archive
            self.assertEqual(len(pool.archive), 2)

    def test_a_deleted_clip_is_removed(self):
        with tempfile.TemporaryDirectory() as folder:
            path = touch(folder, 'A', '01')
            touch(folder, 'A', '02')
            pool = FakePool()
            scanner = FolderScanner(folder, pool, 24)
            scanner.update()
            os.remove(path)
            scanner.update()
            self.assertEqual(pool.removed, ['01'])


class TestFilesThatAreNotRecordings(unittest.TestCase):
    """Clips sorted into a style folder by hand are older than the newest recording: they are archive."""

    def scanner(self, folder: str) -> tuple[FolderScanner, FakePool]:
        touch(folder, 'B', '10')
        touch(folder, 'A', '20')
        touch(folder, 'A', '30')
        pool = FakePool()
        scanner = FolderScanner(folder, pool, 24)
        scanner.update()
        self.assertEqual(pool.style, 'A')
        pool.added.clear()
        return scanner, pool

    def test_an_older_file_in_the_current_style_is_not_loaded(self):
        with tempfile.TemporaryDirectory() as folder:
            scanner, pool = self.scanner(folder)
            touch(folder, 'A', '25')
            scanner.update()
            self.assertEqual(pool.added, [])

    def test_an_older_file_in_the_other_style_does_not_switch(self):
        with tempfile.TemporaryDirectory() as folder:
            scanner, pool = self.scanner(folder)
            for id in ('11', '12', '29'):
                touch(folder, 'B', id)
            scanner.update()
            self.assertEqual(pool.added, [])
            self.assertEqual(pool.style, 'A')

    def test_a_newer_file_in_the_other_style_switches(self):
        with tempfile.TemporaryDirectory() as folder:
            scanner, pool = self.scanner(folder)
            touch(folder, 'B', '31')
            scanner.update()
            self.assertEqual(pool.style, 'B')
            self.assertEqual(pool.added[-1], ('31', 'B'))

    def test_several_new_recordings_in_one_event_all_count(self):
        with tempfile.TemporaryDirectory() as folder:
            scanner, pool = self.scanner(folder)
            touch(folder, 'A', '32')
            touch(folder, 'A', '31')
            scanner.update()
            self.assertEqual(pool.added, [('31', 'A'), ('32', 'A')])
