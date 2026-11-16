"""The view stage's geometry and its depth read across a surface, as plain arrays."""

import unittest

import numpy as np

from modules.flow import (depth_crop, depth_step, edge_aware_depth, fixed_window, masked_depth, place_points,
                          push_pull, pyramid_sizes, view_per_depth_pixel, view_span)
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


class DepthCropTest(unittest.TestCase):
    def crop(self, depth: np.ndarray, x: float, y: float) -> float:
        height, width = depth.shape
        return float(depth_crop(depth, np.array([x / width]), np.array([y / height]), 1.0, 2.0)[0])

    def test_the_depth_pixel_a_point_lies_in_decides_unblended(self) -> None:
        depth = np.array([[1.5, 3.0], [1.5, 3.0]])                 # the player at 1.5 m, the room behind at 3 m
        self.assertEqual(self.crop(depth, 0.9, 1.0), 1.5)          # in the player's pixel, near the room's
        self.assertEqual(self.crop(depth, 1.1, 1.0), 0.0)          # in the room's pixel, behind the box

    def test_no_depth_and_beyond_the_grid_are_0(self) -> None:
        depth = np.array([[1.5, 0.0], [1.5, 0.0]])
        self.assertEqual(self.crop(depth, 1.5, 1.0), 0.0)
        self.assertEqual(self.crop(depth, -0.5, 1.0), 0.0)


class MaskedDepthTest(unittest.TestCase):
    def read(self, depth: np.ndarray, x: float, y: float, mask: float) -> float:
        height, width = depth.shape
        return float(masked_depth(depth, np.array([x / width]), np.array([y / height]), 0.2, np.array([mask]),
                                  1.0, 2.0)[0])

    def test_the_player_s_depth_where_the_mask_is(self) -> None:
        depth = np.array([[1.5, 1.6], [1.5, 1.6]])
        self.assertAlmostEqual(self.read(depth, 1.0, 1.0, mask=0.3), 1.55)     # any mask above 0

    def test_no_mask_or_not_the_player_is_0(self) -> None:
        depth = np.array([[1.5, 3.0], [1.5, 3.0]])                 # the player at 1.5 m, the room behind at 3 m
        self.assertEqual(self.read(depth, 0.8, 1.0, mask=0.0), 0.0)
        self.assertEqual(self.read(depth, 1.2, 1.0, mask=1.0), 0.0)            # the room's, under the mask


class PushPullTest(unittest.TestCase):
    def test_a_depth_is_kept(self) -> None:
        read = np.array([[1.5, 1.6], [1.7, 1.8]])
        np.testing.assert_allclose(push_pull(read, np.ones_like(read)), read)

    def test_a_hole_takes_the_depth_around_it(self) -> None:
        read = np.full((5, 5), 1.5)
        read[2, 2] = 0.0
        self.assertAlmostEqual(float(push_pull(read, np.ones_like(read))[2, 2]), 1.5)

    def test_the_band_takes_the_player_s_depth(self) -> None:
        read = np.zeros((8, 8))
        read[:, :4] = 1.5                                           # the player's depth up to the depth's edge
        mask = np.zeros((8, 8))
        mask[:, :6] = 0.5                                           # the mask two pixels further
        filled = push_pull(read, mask)
        np.testing.assert_allclose(filled[:, 4:6], 1.5)
        self.assertTrue((filled[:, 6:] == 0.0).all())               # none beyond the mask

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
