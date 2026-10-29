"""Tests for the pool's two lists: the last recordings (recent) and the other clips (archive)."""

import os
import unittest

import numpy as np

from apps.smile.player.clip_pool import Clip, ClipPool
from apps.smile.player.tests.test_even_spread import turns


def clip(id: str, style: str = 'A') -> Clip:
    return Clip(id, np.zeros((1, 2, 2, 3), np.uint8), 15.0, style)


def ids(clips: list[Clip]) -> list[str]:
    return [c.id for c in clips]


def path(style: str, id: str) -> str:
    return os.path.join('grins', style, id + '.mp4')


class CountingPool(ClipPool):
    """A pool whose decoder makes an empty clip and counts its calls."""

    def __init__(self, size: int, recent: int) -> None:
        super().__init__(size, (2, 2), recent=recent)
        self.decoded: list[str] = []

    def _decode(self, id: str, path: str, style: str) -> Clip | None:
        self.decoded.append(id)
        return clip(id, style)


class TestLists(unittest.TestCase):

    def test_a_recording_joins_recent_in_time_order(self):
        pool = ClipPool(10, (2, 2), recent=3)
        for id in ('05', '03', '04'):
            pool._insert(clip(id), recent=True)
        self.assertEqual(ids(pool._recent), ['03', '04', '05'])
        self.assertEqual(pool._archive, [])

    def test_one_recording_too_many_moves_the_oldest_to_the_end_of_the_archive(self):
        pool = ClipPool(10, (2, 2), recent=3)
        for id in ('a1', 'a2'):
            pool._insert(clip(id))
        for id in ('01', '02', '03', '04'):
            pool._insert(clip(id), recent=True)
        self.assertEqual(ids(pool._recent), ['02', '03', '04'])
        self.assertEqual(ids(pool._archive), ['a1', 'a2', '01'])

    def test_a_clip_from_disk_joins_the_end_of_the_archive(self):
        pool = ClipPool(10, (2, 2), recent=3)
        for id in ('z', 'b', 'm'):
            pool._insert(clip(id))
        self.assertEqual(ids(pool._archive), ['z', 'b', 'm'])
        self.assertEqual(pool._recent, [])

    def test_over_the_size_the_oldest_of_the_archive_leaves(self):
        pool = ClipPool(4, (2, 2), recent=2)
        for id in ('a1', 'a2'):
            pool._insert(clip(id))
        for id in ('r1', 'r2'):
            pool._insert(clip(id), recent=True)
        pool._insert(clip('a3'))
        self.assertEqual(ids(pool._archive), ['a2', 'a3'])
        self.assertEqual(ids(pool._recent), ['r1', 'r2'])

    def test_the_recent_survive_any_number_of_clips_from_disk(self):
        pool = ClipPool(6, (2, 2), recent=2)
        for id in ('r1', 'r2'):
            pool._insert(clip(id), recent=True)
        for i in range(50):
            pool._insert(clip(f'd{i:02d}'))
        self.assertEqual(ids(pool._recent), ['r1', 'r2'])
        self.assertEqual(ids(pool._archive), ['d46', 'd47', 'd48', 'd49'])

    def test_a_recording_that_leaves_recent_gets_a_full_turn_in_the_archive(self):
        pool = ClipPool(6, (2, 2), recent=2)
        for id in ('r1', 'r2'):
            pool._insert(clip(id), recent=True)
        for i in range(50):                         # a long quiet spell of recycling
            pool._insert(clip(f'd{i:02d}'))
        pool._insert(clip('r3'), recent=True)
        self.assertEqual(ids(pool._archive)[-1], 'r1')      # the newest of the archive, not the first to go
        pool._insert(clip('r4'), recent=True)
        for i in range(2):
            pool._insert(clip(f'e{i}'))
            self.assertIn('r1', pool.ids())
        pool._insert(clip('e2'))
        self.assertNotIn('r1', pool.ids())                  # four clips joined after it: its turn is over
        self.assertIn('r2', pool.ids())

    def test_no_recent_list_puts_recordings_in_the_archive(self):
        pool = ClipPool(3, (2, 2), recent=0)
        for id in ('01', '02'):
            pool._insert(clip(id), recent=True)
        self.assertEqual(pool._recent, [])
        self.assertEqual(ids(pool._archive), ['01', '02'])

    def test_a_pool_below_its_size_drops_nothing(self):
        pool = ClipPool(30, (2, 2), recent=4)
        for i in range(10):
            pool._insert(clip(f'{i:02d}'), recent=i >= 6)
        self.assertEqual(len(pool), 10)

    def test_remove_works_in_both_lists(self):
        pool = ClipPool(6, (2, 2), recent=2)
        pool._insert(clip('a1'))
        pool._insert(clip('r1'), recent=True)
        pool.remove('a1')
        pool.remove('r1')
        self.assertEqual(len(pool), 0)


class TestSwitch(unittest.TestCase):

    def test_what_was_in_the_pool_leaves_first_and_its_last_recordings_last(self):
        pool = ClipPool(4, (2, 2), recent=2)
        pool.switch('A')
        for id in ('a1', 'a2'):
            pool._insert(clip(id))
        for id in ('a3', 'a4'):
            pool._insert(clip(id), recent=True)
        pool.switch('B')
        self.assertEqual(ids(pool._archive), ['a1', 'a2', 'a3', 'a4'])
        self.assertEqual(pool._recent, [])
        pool._insert(clip('b9', 'B'), recent=True)
        self.assertEqual(ids(pool._archive), ['a2', 'a3', 'a4'])
        pool._insert(clip('b1', 'B'))
        pool._insert(clip('b2', 'B'))
        self.assertEqual(ids(pool._archive), ['a4', 'b1', 'b2'])
        pool._insert(clip('b3', 'B'))
        self.assertEqual(ids(pool._archive), ['b1', 'b2', 'b3'])
        self.assertEqual(ids(pool._recent), ['b9'])

    def test_the_clips_loaded_with_a_switch_leave_before_the_clip_that_caused_it(self):
        pool = CountingPool(4, recent=2)
        pool.switch('B')
        pool._load(path('B', 'b9'), 'B', True)          # the switching clip is decoded first
        for id in ('b1', 'b2', 'b3'):
            pool._load(path('B', id), 'B', False)       # the clips from disk after it
        self.assertEqual(pool.decoded[0], 'b9')
        pool._load(path('B', 'c1'), 'B', True)
        pool._load(path('B', 'c2'), 'B', True)          # b9 is no longer among the last recordings
        self.assertEqual(ids(pool._archive), ['b3', 'b9'])      # b1 and b2 went, b9 is the newest of the archive
        pool._load(path('B', 'd1'), 'B', False)
        self.assertEqual(ids(pool._archive), ['b9', 'd1'])

    def test_a_file_of_the_style_the_wall_left_is_not_decoded(self):
        pool = CountingPool(6, recent=2)
        pool.switch('A')
        pool._load(path('A', 'a1'), 'A', True)
        pool.switch('B')
        pool._load(path('A', 'a2'), 'A', True)          # still queued from before the switch
        pool._load(path('B', 'b1'), 'B', True)
        self.assertEqual(pool.decoded, ['a1', 'b1'])
        self.assertEqual(ids(pool._recent), ['b1'])

    def test_a_clip_decoded_while_the_wall_switched_does_not_join(self):
        pool = ClipPool(6, (2, 2), recent=2)
        pool.switch('B')
        pool._insert(clip('a1', 'A'), recent=True)
        self.assertEqual(len(pool), 0)

    def test_a_switch_back_reuses_the_clips_still_in_memory(self):
        pool = CountingPool(6, recent=2)
        pool.switch('A')
        for id in ('a1', 'a2', 'a3', 'a4'):
            pool._load(path('A', id), 'A', False)
        for id in ('a5', 'a6'):
            pool._load(path('A', id), 'A', True)
        pool.switch('B')                                # a style with only two clips
        pool._load(path('B', 'b1'), 'B', True)
        pool._load(path('B', 'b2'), 'B', False)
        self.assertEqual(sorted(pool.ids()), ['a3', 'a4', 'a5', 'a6', 'b1', 'b2'])
        pool.decoded.clear()

        pool.switch('A')                                # back: a7 is new, the others are still there
        for id in ('a3', 'a4', 'a5'):
            pool._load(path('A', id), 'A', False)
        pool._load(path('A', 'a6'), 'A', True)
        pool._load(path('A', 'a7'), 'A', True)
        self.assertEqual(pool.decoded, ['a7'])
        self.assertEqual(ids(pool._recent), ['a6', 'a7'])
        self.assertEqual(len(pool), 6)
        self.assertEqual(sorted(c.id for c in pool.clips()), ['a3', 'a4', 'a5', 'a6', 'a7'])


class TestDemotedRecordingOnTheWall(unittest.TestCase):

    def test_a_recording_kept_long_in_recent_is_shown_like_the_others_after_it_leaves(self):
        pool = ClipPool(30, (2, 2), recent=4)
        for i in range(26):
            pool._insert(clip(f'a{i:02d}'))
        for i in range(4):
            pool._insert(clip(f'r{i}'), recent=True)
        kept = pool._recent[0]
        playing: list[Clip | None] = [None] * 24
        for cell in turns(24, 300):                     # a quiet hour: the recent clips collect plays
            playing[cell] = pool.get_clip(playing[cell])
        self.assertGreater(kept.plays, max(c.plays for c in pool._archive) + 30)

        for i in range(4):                              # four visitors: the old four move to the archive
            pool._insert(clip(f's{i}'), recent=True)
        self.assertIn(kept, pool._archive)
        before = {c.id: c.plays for c in pool._archive}
        for cell in turns(24, 40):
            playing[cell] = pool.get_clip(playing[cell])
        gained = {c.id: c.plays - before[c.id] for c in pool._archive}
        others = [n for id, n in gained.items() if id != kept.id]
        self.assertGreaterEqual(gained[kept.id], min(others) - 3)       # without the levelling it would get none
        self.assertGreater(gained[kept.id], 20)


class TestFewClips(unittest.TestCase):

    def test_one_clip_fills_every_cell_and_repeats(self):
        pool = ClipPool(30, (2, 2), recent=4)
        only = clip('01')
        pool._insert(only, recent=True)
        playing: list[Clip | None] = [None] * 24
        for cell in turns(24, 3):
            playing[cell] = pool.get_clip(playing[cell])
            self.assertIs(playing[cell], only)
        self.assertEqual(only.active, 24)

    def test_fewer_recordings_than_the_recent_list_share_the_wall(self):
        pool = ClipPool(30, (2, 2), recent=4)
        for id in ('01', '02', '03'):
            pool._insert(clip(id), recent=True)
        playing: list[Clip | None] = [None] * 24
        for cell in turns(24, 20):
            playing[cell] = pool.get_clip(playing[cell])
        actives = [c.active for c in pool.clips()]
        self.assertEqual(sum(actives), 24)
        self.assertGreaterEqual(min(actives), 4)        # 8 each, a few less or more while cells change over
        self.assertLessEqual(max(actives), 12)
