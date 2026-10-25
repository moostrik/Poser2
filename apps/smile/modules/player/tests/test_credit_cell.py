"""Tests for the cell with the artist's credit: it stays for one play, then takes the next cell that starts (no GL)."""

import unittest

from modules.player.Render import next_credit_cell


class TestCreditCell(unittest.TestCase):
    def test_takes_a_cell_where_a_clip_starts(self):
        self.assertEqual(next_credit_cell(None, None, []), (None, None))
        self.assertEqual(next_credit_cell(None, None, ['4']), ('4', None))

    def test_stays_while_its_clip_plays(self):
        self.assertEqual(next_credit_cell('4', None, ['1', '2']), ('4', None))

    def test_leaves_when_its_cell_starts_the_next_clip(self):
        self.assertEqual(next_credit_cell('4', None, ['4']), (None, '4'))
        self.assertEqual(next_credit_cell('4', None, ['4', '7']), ('7', '4'))

    def test_never_back_to_the_cell_it_left(self):
        self.assertEqual(next_credit_cell(None, '4', ['4']), (None, '4'))
        self.assertEqual(next_credit_cell(None, '4', ['4', '2']), ('2', '4'))


if __name__ == '__main__':
    unittest.main()
