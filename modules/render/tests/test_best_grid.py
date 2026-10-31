"""The grid that shows a number of equal cells largest in a window."""

import unittest

from modules.render import best_grid

PORTRAIT: float = 1080 / 1920


class BestGridTest(unittest.TestCase):
    def test_two_portrait_outputs_side_by_side_in_a_wide_window(self) -> None:
        self.assertEqual(best_grid(2, PORTRAIT, 1920, 1080), (2, 1))

    def test_two_portrait_outputs_stacked_in_a_tall_narrow_window(self) -> None:
        self.assertEqual(best_grid(2, PORTRAIT, 800, 2400), (1, 2))

    def test_two_landscape_outputs_stacked_in_a_square_window(self) -> None:
        self.assertEqual(best_grid(2, 16 / 9, 1000, 1000), (1, 2))

    def test_one_cell_is_one_by_one(self) -> None:
        self.assertEqual(best_grid(1, PORTRAIT, 640, 480), (1, 1))


if __name__ == '__main__':
    unittest.main()
