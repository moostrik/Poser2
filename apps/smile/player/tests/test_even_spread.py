"""Tests for how the clips are spread over the cells: every clip about equally, none back to back in a cell."""

import random
import unittest

import numpy as np

from apps.smile.player.clip_pool import Clip, ClipPool, choose


def clip(id: str, plays: int = 0, active: int = 0) -> Clip:
    return Clip(id, np.zeros((1, 2, 2, 3), np.uint8), 15.0, 'A', plays, active, [active, 0, 0])


def pool_of(count: int, size: int | None = None) -> ClipPool:
    pool = ClipPool(size or count, (2, 2))
    for i in range(count):
        pool._insert(clip(f'{i:03d}'))
    return pool


def turns(cells: int, rounds: int):
    """The cells finishing their plays: every cell once per round (all plays last about as long), in a
    different order each round (the speeds differ a little)."""
    for _ in range(rounds):
        yield from random.sample(range(cells), cells)


def run_cells(pool: ClipPool, cells: int, rounds: int) -> list[Clip | None]:
    """Each cell takes its next clip, round after round; returns what each cell plays at the end."""
    playing: list[Clip | None] = [None] * cells
    for cell in turns(cells, rounds):
        playing[cell] = pool.get_clip(playing[cell])
    return playing


class TestChoose(unittest.TestCase):

    def test_a_clip_on_no_cell_goes_before_one_on_a_cell(self):
        clips = [clip('a', plays=0, active=1), clip('b', plays=9, active=0)]
        self.assertEqual({choose(clips).id for _ in range(30)}, {'b'})

    def test_among_equals_the_least_played(self):
        clips = [clip('a', plays=3), clip('b', plays=1), clip('c', plays=2)]
        self.assertEqual({choose(clips).id for _ in range(30)}, {'b'})

    def test_random_among_full_equals(self):
        clips = [clip('a'), clip('b'), clip('c')]
        self.assertEqual({choose(clips).id for _ in range(200)}, {'a', 'b', 'c'})

    def test_never_the_clip_just_played_when_there_is_another(self):
        a, b = clip('a', plays=0), clip('b', plays=50, active=3)
        self.assertEqual({choose([a, b], exclude=a).id for _ in range(30)}, {'b'})

    def test_the_only_clip_repeats(self):
        a = clip('a')
        self.assertIs(choose([a], exclude=a), a)

    def test_empty(self):
        self.assertIsNone(choose([]))


class TestSpread(unittest.TestCase):

    def check_counts(self, pool: ClipPool, playing: list[Clip | None]) -> None:
        """The pool's bookkeeping matches what the cells really play."""
        for c in pool.clips():
            self.assertEqual(c.active, sum(1 for p in playing if p is c), c.id)

    def test_as_many_clips_as_cells(self):
        pool = pool_of(24)
        playing: list[Clip | None] = [None] * 24
        for step, cell in enumerate(turns(24, 120)):
            playing[cell] = pool.get_clip(playing[cell])
            if step > 200:
                shown, most = pool.spread()
                self.assertLessEqual(most, 2)
                self.assertGreaterEqual(shown, 22)
        self.check_counts(pool, playing)
        plays = [c.plays for c in pool.clips()]
        self.assertLessEqual(max(plays) - min(plays), 6)        # 120 plays each over the run, within a few

    def test_more_clips_than_cells_shows_none_twice_and_all_in_turn(self):
        pool = pool_of(30)
        playing: list[Clip | None] = [None] * 24
        seen: set[str] = set()
        for step, cell in enumerate(turns(24, 120)):
            playing[cell] = pool.get_clip(playing[cell])
            seen.add(playing[cell].id)
            if step > 200:
                self.assertEqual(pool.spread(), (24, 1))
        self.check_counts(pool, playing)
        self.assertEqual(len(seen), 30)
        plays = [c.plays for c in pool.clips()]
        self.assertLessEqual(max(plays) - min(plays), 5)        # 96 plays each over the run, within a few

    def test_fewer_clips_than_cells_share_the_cells(self):
        pool = pool_of(6)
        playing = run_cells(pool, 24, 80)
        self.check_counts(pool, playing)
        actives = [c.active for c in pool.clips()]
        self.assertEqual(sum(actives), 24)
        # 4 each; a clip is one or two short for a moment when its cells finish back to back,
        # since a cell never takes back the clip it just played
        self.assertGreaterEqual(min(actives), 2)
        self.assertLessEqual(max(actives), 6)

    def test_no_clip_twice_in_a_row_in_a_cell(self):
        pool = pool_of(5)
        playing: list[Clip | None] = [None] * 3
        for cell in turns(3, 150):
            before = playing[cell]
            playing[cell] = pool.get_clip(before)
            self.assertIsNot(playing[cell], before)

    def test_a_new_clip_is_shown_within_a_round(self):
        pool = pool_of(30, size=31)
        playing = run_cells(pool, 24, 40)
        new = clip('new')
        pool._insert(new)
        # it joins level with the least played, so each other clip can be taken before it once at most
        for step, cell in enumerate(turns(24, 2)):
            playing[cell] = pool.get_clip(playing[cell])
            if playing[cell] is new:
                break
        self.assertEqual(new.active, 1)
        self.assertLess(step, 31)

    def test_a_new_clip_is_level_with_the_rest_afterwards(self):
        pool = pool_of(30, size=31)
        playing = run_cells(pool, 24, 40)
        pool._insert(clip('new'))
        taken = 0
        for cell in turns(24, 31):      # 744 plays over 31 clips: 24 each
            playing[cell] = pool.get_clip(playing[cell])
            taken += playing[cell].id == 'new'
        self.assertGreaterEqual(taken, 22)
        self.assertLessEqual(taken, 26)

    def test_the_starter_and_a_stopping_cell_count_too(self):
        pool = pool_of(3)
        first = pool.clips()[0]
        pool.use(first)
        self.assertEqual(pool.spread(), (1, 1))
        self.assertIsNot(pool.get_clip(), first)        # a second cell takes a clip that is on no cell
        pool.release(first)
        self.assertEqual(pool.spread(), (1, 1))
