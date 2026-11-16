"""The view stage's geometry and the fill's pyramid sizes, as plain arrays."""

import unittest

import numpy as np

from modules.flow import depth_step, fixed_window, place_points, pyramid_sizes, view_per_depth_pixel, view_span
from modules.utils import Rect


class PyramidSizesTest(unittest.TestCase):
    def test_sizes_down_to_one(self) -> None:
        self.assertEqual(pyramid_sizes(800, 448)[-1], (1, 1))
        self.assertEqual(pyramid_sizes(5, 3), [(3, 2), (2, 1), (1, 1)])


class ViewWindowTest(unittest.TestCase):
    def test_the_fixed_window_keeps_the_view_s_shape(self) -> None:
        window = fixed_window(0.5, 0.5, 1.0, 448 / 800, 992 / 800)
        self.assertAlmostEqual(window.width * 992 / (window.height * 800), 448 / 800)
        self.assertAlmostEqual(window.x + window.width / 2.0, 0.5)

    def test_spans_and_steps(self) -> None:
        window = Rect(0.25, 0.0, 0.5, 1.0)
        self.assertAlmostEqual(view_span(window, 90.0), 1.0)       # half of a field 2 m wide at 1 m
        self.assertEqual(view_span(window, 0.0), 0.0)
        self.assertEqual(depth_step(window, (100, 50)), (1.0 / 100 / 0.5, 1.0 / 50))
        self.assertEqual(depth_step(window, (0, 0)), (0.0, 0.0))
        self.assertAlmostEqual(view_per_depth_pixel(window, 300, 800), 800 / 300)

    def test_points_are_placed_in_the_window(self) -> None:
        x, y = place_points(np.array([0.0, 1.0]), np.array([0.0, 1.0]), Rect(0.25, 0.0, 0.5, 1.0), Rect(0.25, 0.0, 0.5, 1.0))
        np.testing.assert_allclose(x, [0.0, 1.0])
        np.testing.assert_allclose(y, [0.0, 1.0])


if __name__ == '__main__':
    unittest.main()
