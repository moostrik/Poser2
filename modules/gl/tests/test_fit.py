"""`fit` and `fill` place a source in a destination, centered."""

import unittest

from modules.gl import fill, fit


class FitTest(unittest.TestCase):
    def test_fit_shows_all_of_a_wide_source_in_a_tall_target(self) -> None:
        self.assertEqual(fit(200, 100, 100, 100), (0.0, 25.0, 100.0, 50.0))

    def test_fit_shows_all_of_a_tall_source_in_a_wide_target(self) -> None:
        self.assertEqual(fit(100, 200, 100, 100), (25.0, 0.0, 50.0, 100.0))

    def test_fill_covers_the_target_and_crops_the_rest(self) -> None:
        self.assertEqual(fill(200, 100, 100, 100), (-50.0, 0.0, 200.0, 100.0))
        self.assertEqual(fill(100, 200, 100, 100), (0.0, -50.0, 100.0, 200.0))

    def test_the_same_shape_fills_the_target_either_way(self) -> None:
        self.assertEqual(fit(16, 9, 1920, 1080), (0.0, 0.0, 1920.0, 1080.0))
        self.assertEqual(fill(16, 9, 1920, 1080), (0.0, 0.0, 1920.0, 1080.0))


if __name__ == '__main__':
    unittest.main()
