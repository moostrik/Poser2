"""The view stage's geometry and its depth read across a surface, as plain arrays."""

import unittest

import numpy as np

from modules.flow import (depth_step, edge_aware_depth, fixed_window, joint_bilateral_depth, mask_sided_depth, place_points, texture_roi,
                          view_per_depth_pixel, view_span)
from modules.utils import Rect


def at(depth: np.ndarray, x: float, y: float, tolerance: float = 0.2) -> float:
    """The read at a point given in the depth's pixels (x right, y down)."""
    height, width = depth.shape
    return float(edge_aware_depth(depth, np.array([x / width]), np.array([y / height]), tolerance)[0])


class EdgeAwareDepthTest(unittest.TestCase):
    def test_within_a_surface_the_depth_is_interpolated(self) -> None:
        depth = np.array([[1.0, 1.1], [1.0, 1.1]])
        self.assertAlmostEqual(at(depth, 1.0, 1.0), 1.05)          # halfway between the pixel centers
        self.assertAlmostEqual(at(depth, 0.5, 0.5), 1.0)           # on a center: the pixel itself

    def test_across_an_edge_the_nearest_is_taken(self) -> None:
        depth = np.array([[1.0, 3.0], [1.0, 3.0]])                 # 2 m apart, past the tolerance
        self.assertEqual(at(depth, 0.9, 1.0), 1.0)
        self.assertEqual(at(depth, 1.1, 1.0), 3.0)

    def test_no_depth_is_never_blended(self) -> None:
        depth = np.array([[1.0, 0.0], [1.0, 1.0]])
        self.assertEqual(at(depth, 0.9, 0.9), 1.0)                 # the nearest, a depth
        self.assertEqual(at(depth, 1.1, 0.9), 0.0)                 # the nearest, none
        self.assertEqual(at(depth, 1.9, 1.9), 1.0)

    def test_beyond_the_grid_is_no_depth(self) -> None:
        depth = np.ones((2, 2))
        self.assertEqual(at(depth, 0.1, 0.1), 1.0)                 # a corner outside: the nearest, inside
        self.assertEqual(at(depth, -0.5, 1.0), 0.0)


class MaskSidedDepthTest(unittest.TestCase):
    def read(self, depth: np.ndarray, x: float, y: float, inside: bool, reach: float = 2.0) -> float:
        height, width = depth.shape
        return float(mask_sided_depth(depth, np.array([x / width]), np.array([y / height]), 0.2, np.array([inside]),
                                      1.0, 2.0, reach)[0])

    def test_at_an_edge_the_mask_gives_the_side(self) -> None:
        depth = np.array([[1.5, 3.0], [1.5, 3.0]])                 # the player at 1.5 m, the room behind at 3 m
        self.assertEqual(self.read(depth, 1.2, 1.0, inside=True), 1.5)     # nearer the room, the mask the player's
        self.assertEqual(self.read(depth, 0.8, 1.0, inside=False), 3.0)    # nearer the player, the mask the room's

    def test_on_its_side_as_surface(self) -> None:
        depth = np.array([[1.5, 1.6], [1.5, 1.6]])
        self.assertAlmostEqual(self.read(depth, 1.0, 1.0, inside=True), 1.55)

    def test_outside_no_depth_is_the_other_side(self) -> None:
        depth = np.array([[1.5, 0.0], [1.5, 0.0]])
        self.assertEqual(self.read(depth, 0.8, 1.0, inside=False), 0.0)

    def test_beyond_reach_as_surface(self) -> None:
        depth = np.array([[1.5, 3.0, 3.0, 3.0, 3.0]] * 2)          # the player's pixel 3 depth pixels off
        self.assertEqual(self.read(depth, 3.5, 1.0, inside=True), 3.0)
        self.assertEqual(self.read(depth, 3.5, 1.0, inside=True, reach=4.0), 1.5)


class JointBilateralDepthTest(unittest.TestCase):
    def read(self, depth: np.ndarray, image: np.ndarray, x: float, y: float) -> float:
        height, width = depth.shape
        return float(joint_bilateral_depth(depth, image, np.array([x / width]), np.array([y / height]), 0.1)[0])

    def test_a_flat_image_is_a_gaussian_of_the_depth(self) -> None:
        depth = np.tile([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (6, 1))
        image = np.full((6, 6, 3), 0.5)
        self.assertAlmostEqual(self.read(depth, image, 3.0, 3.0), 3.5, places=4)   # halfway: the 5 × 5 a column off
        self.assertAlmostEqual(self.read(depth, image, 2.5, 3.0), 3.0)     # on a center: symmetric around it

    def test_the_image_s_edge_keeps_the_depths_apart(self) -> None:
        depth = np.tile([1.5, 1.5, 1.5, 3.0, 3.0, 3.0], (6, 1))
        image = np.zeros((6, 6, 3))
        image[:, 3:] = 1.0
        self.assertAlmostEqual(self.read(depth, image, 2.9, 3.0), 1.5, places=3)
        self.assertAlmostEqual(self.read(depth, image, 3.1, 3.0), 3.0, places=3)

    def test_no_depth_is_left_out(self) -> None:
        depth = np.ones((6, 6))
        depth[:, 3] = 0.0
        image = np.full((6, 6, 3), 0.5)
        self.assertAlmostEqual(self.read(depth, image, 3.5, 3.0), 1.0)


class ViewWindowTest(unittest.TestCase):
    def test_the_fixed_window_keeps_the_view_s_shape(self) -> None:
        window = fixed_window(0.5, 0.5, 1.0, 448 / 800, 992 / 800)
        self.assertAlmostEqual(window.width * 992 / (window.height * 800), 448 / 800)
        self.assertAlmostEqual(window.x + window.width / 2.0, 0.5)

    def test_the_texture_roi_runs_rows_up(self) -> None:
        roi = texture_roi(Rect(0.1, 0.2, 0.3, 0.4))
        self.assertAlmostEqual(roi.y, 0.4)

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
