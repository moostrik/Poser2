"""Tests for the order in which the loaded clips start the cells."""

import unittest

import numpy as np

from apps.smile.player.ClipPool import Clip, ClipPool, next_start
from apps.smile.player.ImgLoader import play_rate


def clip(id: str) -> Clip:
    return Clip(id, np.zeros((1, 2, 2, 3), np.uint8), 15.0)


class TestNextStart(unittest.TestCase):

    def test_first_in_load_order(self):
        clips = [clip('c'), clip('a'), clip('b')]
        self.assertEqual(next_start(clips, set()).id, 'c')

    def test_skips_started(self):
        clips = [clip('c'), clip('a'), clip('b')]
        self.assertEqual(next_start(clips, {'c'}).id, 'a')
        self.assertEqual(next_start(clips, {'c', 'a'}).id, 'b')

    def test_none_when_all_started_or_empty(self):
        self.assertIsNone(next_start([clip('a')], {'a'}))
        self.assertIsNone(next_start([], set()))

    def test_new_clip_is_taken_before_going_round(self):
        clips = [clip('a'), clip('b')]
        started = {'a', 'b'}
        clips.append(clip('new'))
        self.assertEqual(next_start(clips, started).id, 'new')


class TestPoolIdle(unittest.TestCase):

    def test_idle_until_a_file_is_added(self):
        pool = ClipPool(4, (2, 2))
        self.assertTrue(pool.idle)
        pool.add('missing.mp4')
        self.assertFalse(pool.idle)

    def test_use_counts_a_play_and_a_cell(self):
        pool = ClipPool(4, (2, 2))
        c = clip('a')
        for _ in range(3):
            pool.use(c)
        self.assertEqual((c.plays, c.active), (3, 3))
        pool.release(c)
        self.assertEqual((c.plays, c.active), (3, 2))


class TestPlayRate(unittest.TestCase):

    def test_no_spread_is_the_given_rate(self):
        self.assertEqual(play_rate(20.0, 13.0, 0.0), 20.0)

    def test_zero_fps_is_the_clip_rate(self):
        self.assertEqual(play_rate(0.0, 13.0, 0.0), 13.0)

    def test_spread_stays_within_its_fraction_and_varies(self):
        rates = [play_rate(20.0, 13.0, 0.1) for _ in range(500)]
        self.assertGreaterEqual(min(rates), 18.0)
        self.assertLessEqual(max(rates), 22.0)
        self.assertGreater(max(rates) - min(rates), 2.0)

    def test_spread_is_clamped_so_the_rate_stays_positive(self):
        self.assertGreater(min(play_rate(20.0, 13.0, 5.0) for _ in range(500)), 0.0)
