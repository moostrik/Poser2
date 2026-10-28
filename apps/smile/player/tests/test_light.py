"""Tests for the cells the laughter test lights: the bookkeeping the render does each frame (no GL)."""

import unittest

from apps.smile.player.Render import lit_cells


class TestLitCells(unittest.TestCase):
    def test_lit_until_their_time_then_dropped(self):
        lit = {'3': 12.0, '7': 10.5}
        self.assertEqual(lit_cells(lit, 10.0), ['3', '7'])
        self.assertEqual(lit_cells(lit, 10.5), ['3'])       # 7 ended
        self.assertEqual(lit, {'3': 12.0})
        self.assertEqual(lit_cells(lit, 12.0), [])
        self.assertEqual(lit, {})

    def test_a_cell_lit_again_keeps_its_place(self):
        lit = {'3': 12.0}
        lit['3'] = 20.0
        self.assertEqual(lit_cells(lit, 15.0), ['3'])


if __name__ == '__main__':
    unittest.main()
