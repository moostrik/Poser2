"""Tests for the players following SMILE's style: which clips the scanner loads and which the pool keeps."""

import os
import tempfile
import unittest

import numpy as np

from modules.player.ClipPool import Clip, ClipPool, evict_id
from modules.player.FolderScanner import FolderScanner, clip_style, find_clips, list_clips, pick_clips


def clip(id: str, style: str) -> Clip:
    return Clip(id, np.zeros((1, 2, 2, 3), np.uint8), 15.0, style)


def touch(folder: str, style: str, id: str) -> str:
    os.makedirs(os.path.join(folder, style), exist_ok=True)
    path = os.path.join(folder, style, id + '.mp4')
    open(path, 'w').close()
    return path


class FakePool:
    def __init__(self) -> None:
        self.added: list[tuple[str, str]] = []      # (id, style) in the order they were queued
        self.removed: list[str] = []
        self.style: str | None = None

    def add(self, path: str, style: str = '') -> None:
        self.added.append((os.path.splitext(os.path.basename(path))[0], style))

    def remove(self, id: str) -> None:
        self.removed.append(id)

    def set_style(self, style: str | None) -> None:
        self.style = style


class TestClipFiles(unittest.TestCase):

    def test_style_is_the_folder(self):
        self.assertEqual(clip_style(os.path.join('output', 'grins', 'B', '2026_10_01_12_00_00.mp4')), 'B')

    def test_find_clips_ignores_files_without_a_style_folder(self):
        with tempfile.TemporaryDirectory() as folder:
            touch(folder, 'A', '01')
            touch(folder, 'B', '02')
            open(os.path.join(folder, 'old_A.mp4'), 'w').close()
            self.assertEqual(sorted(find_clips(folder)), ['01', '02'])

    def test_list_clips_is_the_newest_of_all_styles(self):
        with tempfile.TemporaryDirectory() as folder:
            for id, style in (('01', 'A'), ('02', 'B'), ('03', 'A'), ('04', 'B')):
                touch(folder, style, id)
            self.assertEqual(list(list_clips(folder, 3)), ['04', '03', '02'])

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
        pool.set_style(style)
        for c in clips:
            pool._clips[c.id] = c
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

    def test_eviction_takes_the_other_style_first(self):
        clips = {c.id: c for c in (clip('01', 'B'), clip('05', 'A'), clip('09', 'A'), clip('10', 'B'))}
        self.assertEqual(evict_id(clips, 'B'), '05')

    def test_eviction_takes_the_oldest_of_the_current_style_last(self):
        clips = {c.id: c for c in (clip('03', 'B'), clip('01', 'B'), clip('10', 'B'))}
        self.assertEqual(evict_id(clips, 'B'), '01')
        self.assertEqual(evict_id(clips, None), '01')

    def test_insert_over_the_size_drops_the_other_style_and_keeps_the_new_clip(self):
        pool = self.pool('B', [clip('01', 'A'), clip('02', 'A'), clip('03', 'A'), clip('04', 'A')])
        pool._insert(clip('05', 'B'))
        self.assertEqual(sorted(c.id for c in pool._clips.values()), ['02', '03', '04', '05'])


class TestScannerFollowsStyle(unittest.TestCase):

    def test_startup_takes_the_style_of_the_newest_clip(self):
        with tempfile.TemporaryDirectory() as folder:
            for id in ('01', '02', '03'):
                touch(folder, 'A', id)
            touch(folder, 'B', '04')
            touch(folder, 'B', '05')
            pool = FakePool()
            scanner = FolderScanner(folder, pool, 24, follow_style=True)
            scanner.update()
            self.assertEqual(pool.style, 'B')
            self.assertEqual(sorted(pool.added), [('04', 'B'), ('05', 'B')])
            self.assertEqual(pool.added[-1], ('05', 'B'))      # the newest is queued last: decoded first

    def test_a_clip_of_the_current_style_is_added(self):
        with tempfile.TemporaryDirectory() as folder:
            touch(folder, 'A', '01')
            pool = FakePool()
            scanner = FolderScanner(folder, pool, 24, follow_style=True)
            scanner.update()
            touch(folder, 'A', '02')
            scanner.update()
            self.assertEqual(pool.added, [('01', 'A'), ('02', 'A')])
            self.assertEqual(pool.style, 'A')

    def test_a_clip_of_the_other_style_switches(self):
        with tempfile.TemporaryDirectory() as folder:
            for id in ('01', '02', '03', '04'):
                touch(folder, 'B', id)      # an earlier B period
            for id in ('05', '06'):
                touch(folder, 'A', id)
            pool = FakePool()
            scanner = FolderScanner(folder, pool, 3, follow_style=True)
            scanner.update()
            self.assertEqual(pool.style, 'A')
            pool.added.clear()

            touch(folder, 'B', '07')
            scanner.update()
            self.assertEqual(pool.style, 'B')
            self.assertEqual(len(pool.added), 3)                # two older B clips and the new one
            self.assertEqual(pool.added[-1], ('07', 'B'))       # queued last: decoded first
            self.assertTrue(all(style == 'B' for _, style in pool.added))
            self.assertTrue({id for id, _ in pool.added[:-1]} <= {'01', '02', '03', '04'})

    def test_a_deleted_clip_is_removed(self):
        with tempfile.TemporaryDirectory() as folder:
            path = touch(folder, 'A', '01')
            touch(folder, 'A', '02')
            pool = FakePool()
            scanner = FolderScanner(folder, pool, 24, follow_style=True)
            scanner.update()
            os.remove(path)
            scanner.update()
            self.assertEqual(pool.removed, ['01'])

    def test_without_the_option_the_newest_of_all_styles_are_loaded(self):
        with tempfile.TemporaryDirectory() as folder:
            touch(folder, 'A', '01')
            touch(folder, 'B', '02')
            touch(folder, 'A', '03')
            pool = FakePool()
            scanner = FolderScanner(folder, pool, 2)
            scanner.update()
            self.assertEqual(pool.added, [('02', 'B'), ('03', 'A')])
            self.assertIsNone(pool.style)
