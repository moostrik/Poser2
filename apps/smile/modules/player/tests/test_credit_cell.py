"""Tests for the cell with the artist's credit: it is always on one cell and moves on (no GL)."""

import unittest

from modules.player.Render import PlayerRender, next_credit_cell


class TestCreditCell(unittest.TestCase):
    def test_never_the_cell_it_is_on(self):
        ids = [str(i) for i in range(4)]
        for _ in range(50):
            self.assertNotEqual(next_credit_cell(ids, '2'), '2')

    def test_a_single_cell_keeps_it(self):
        self.assertEqual(next_credit_cell(['0'], '0'), '0')

    def render(self, every: float):
        render = PlayerRender.__new__(PlayerRender)     # the bookkeeping only: no window
        render._ids = ['0', '1', '2']
        render.credit_every = every
        render._credit_cell = None
        render._credit_moved = 0.0
        return render

    def test_on_a_cell_from_the_start_then_moves_when_due(self):
        render = self.render(30.0)
        took = render._move_credit(100.0)
        self.assertEqual(took, [render._credit_cell])
        first = render._credit_cell
        self.assertEqual(render._move_credit(129.0), [])
        moved = render._move_credit(130.0)
        self.assertEqual(moved[0], first)               # both cells get the white frame
        self.assertNotEqual(render._credit_cell, first)

    def test_no_credit(self):
        render = self.render(0.0)
        self.assertEqual(render._move_credit(100.0), [])
        self.assertIsNone(render._credit_cell)


if __name__ == '__main__':
    unittest.main()
