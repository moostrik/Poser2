"""Tests for the order in which the loaded clips start the cells."""

import unittest

import numpy as np

from player.ClipPool import Clip, ClipPool, next_start


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

    def test_use_counts_up_to_the_maximum(self):
        pool = ClipPool(4, (2, 2), max_usage_count=2)
        c = clip('a')
        for _ in range(3):
            pool.use(c)
        self.assertEqual(c.usage_count, 2)
