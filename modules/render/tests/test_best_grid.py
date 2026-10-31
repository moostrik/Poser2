"""The grid that shows a number of cells largest in a window."""

import unittest

from modules.render import best_grid, grid_cell_aspect_ratio

PORTRAIT: float = 1080 / 1920
WIDE: float = 16 / 9


class BestGridTest(unittest.TestCase):
    def test_two_portrait_outputs_side_by_side_in_a_wide_window(self) -> None:
        self.assertEqual(best_grid([PORTRAIT, PORTRAIT], 1920, 1080), (2, 1))

    def test_two_portrait_outputs_stacked_in_a_tall_narrow_window(self) -> None:
        self.assertEqual(best_grid([PORTRAIT, PORTRAIT], 800, 2400), (1, 2))

    def test_two_landscape_outputs_stacked_in_a_square_window(self) -> None:
        self.assertEqual(best_grid([WIDE, WIDE], 1000, 1000), (1, 2))

    def test_two_wide_cells_stacked_in_a_tall_window(self) -> None:
        self.assertEqual(best_grid([WIDE, WIDE], 1080, 1920), (1, 2))

    def test_two_wide_cells_side_by_side_in_a_very_wide_window(self) -> None:
        self.assertEqual(best_grid([WIDE, WIDE], 3840, 1080), (2, 1))

    def test_a_wide_and_a_portrait_cell_compare_by_total_area(self) -> None:
        self.assertEqual(best_grid([WIDE, PORTRAIT], 1200, 1000), (2, 1))

    def test_one_cell_is_one_by_one(self) -> None:
        self.assertEqual(best_grid([PORTRAIT], 640, 480), (1, 1))


class GridCellAspectRatioTest(unittest.TestCase):
    def test_equal_cells_keep_their_aspect_ratio(self) -> None:
        self.assertAlmostEqual(grid_cell_aspect_ratio([PORTRAIT, PORTRAIT], 2, 1, 1920, 1080), PORTRAIT)

    def test_mixed_cells_take_the_widest_width_and_the_tallest_height(self) -> None:
        self.assertAlmostEqual(grid_cell_aspect_ratio([WIDE, PORTRAIT], 2, 1, 1200, 1000), 600 / 1000)


if __name__ == '__main__':
    unittest.main()
