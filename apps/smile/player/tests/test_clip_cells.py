"""Tests for the cells' plays, stepped directly (no thread, no window, no OSC): the bang reports the cell a play
starts in, which the Max patch turns into the place of the laugh, and each cell keeps its own clock."""

import unittest
from types import SimpleNamespace
from typing import cast

import numpy as np

from apps.smile.player.clip_cells import ClipCells
from apps.smile.player.clip_pool import Clip, ClipPool
from apps.smile.player.settings import WallsSettings


def cells(ids: list[int], fps: float) -> ClipCells:
    return ClipCells([(id, 0) for id in ids], ClipPool(1, (512, 640)), cast(WallsSettings, SimpleNamespace(fps=fps, spread=0.0)))


class TestClipCells(unittest.TestCase):
    def test_a_play_reports_the_cell_it_starts_in(self):
        player = cells([7], 100.0)
        played: list[int] = []
        player.add_play_callback(played.append)
        player.assign(7, Clip('x', np.zeros((2, 4, 4, 3), dtype=np.uint8), 20.0))
        player.step(player.cells[7].end_time)
        self.assertEqual(played, [7])

    def test_each_cell_on_its_own_clock(self):
        player = cells([1, 2], 10.0)
        shown: list[tuple[int, bool]] = []
        player.add_image_callback(lambda id, image, first: shown.append((id, first)))
        clip = Clip('x', np.zeros((3, 4, 4, 3), dtype=np.uint8), 20.0)
        player.assign(1, clip)
        start: float = player.cells[1].end_time
        next_due = player.step(start)
        self.assertEqual(shown, [(1, True)])
        self.assertAlmostEqual(next_due, start + 0.1)         # 10 fps
        player.assign(2, clip)
        second: float = player.cells[2].end_time
        player.step(second)
        self.assertEqual(shown, [(1, True), (2, True)])
        player.step(start + 0.1)                                # cell 1's second frame, cell 2's not yet due
        self.assertEqual(shown[-1], (1, False))
        self.assertEqual(len(shown), 3)
        player.step(second + 0.1)
        self.assertEqual(shown[-1], (2, False))


if __name__ == '__main__':
    unittest.main()
