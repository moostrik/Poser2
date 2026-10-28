"""Tests for what a cell reports when a play starts: its own index, which the Max patch turns into the
place of the laugh (no window, no OSC: the loader is driven directly)."""

import unittest
from types import SimpleNamespace

import numpy as np

from modules.player.ClipPool import Clip, ClipPool
from modules.player.ImgLoader import LoadImages


class TestBang(unittest.TestCase):
    def test_a_play_reports_the_cell_it_starts_in(self):
        pool = ClipPool(1, (512, 640))
        cells: list[int] = []
        loader = LoadImages('7', pool, SimpleNamespace(fps=100.0, spread=0.0))
        loader.add_loader_callback(cells.append)
        # the first frame ends the play
        loader.add_image_callback(lambda id, image, initial: loader._stop_event.set())
        clip = Clip('x', np.zeros((2, 4, 4, 3), dtype=np.uint8), 20.0)
        loader._play(clip)
        self.assertEqual(cells, [7])


if __name__ == '__main__':
    unittest.main()
