"""The Lucas–Kanade flow's numpy references: it finds how far an image moved and gives that as a
velocity. Also the trail bridge's shaping of a velocity. Its depth is tested in test_depth_velocity."""

import unittest

import numpy as np

from modules.flow import displacement_velocity, lucas_kanade, shape_velocity, threshold_displacement

SIZE: int = 64


def pattern(dx: float = 0.0, dy: float = 0.0) -> np.ndarray:
    """A smooth luminance pattern (0 to 1), moved by (dx, dy) pixels: its value at x is the unmoved one at x − d."""
    rows, columns = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
    x, y = columns - dx, rows - dy
    return 0.5 + 0.2 * np.sin(x / 5.0) * np.cos(y / 7.0) + 0.15 * np.sin((x + y) / 9.0) + 0.1 * np.cos(x / 13.0 - y / 6.0)


def interior(field: np.ndarray, margin: int = 12) -> np.ndarray:
    return field[margin:-margin, margin:-margin]


class LucasKanadeTest(unittest.TestCase):
    def test_a_one_pixel_shift_is_found(self) -> None:
        d = lucas_kanade(pattern(), pattern(1.0, 0.0))
        self.assertAlmostEqual(float(np.median(interior(d[..., 0]))), 1.0, delta=0.1)
        self.assertAlmostEqual(float(np.median(interior(d[..., 1]))), 0.0, delta=0.1)

    def test_a_larger_shift_is_found_through_the_mip_levels(self) -> None:
        d = lucas_kanade(pattern(), pattern(4.0, -2.0), levels=3)
        self.assertAlmostEqual(float(np.median(interior(d[..., 0]))), 4.0, delta=0.5)
        self.assertAlmostEqual(float(np.median(interior(d[..., 1]))), -2.0, delta=0.5)

    def test_no_motion_is_no_displacement(self) -> None:
        d = lucas_kanade(pattern(), pattern())
        self.assertLess(float(np.abs(d).max()), 1e-6)

    def test_a_flat_image_solves_to_no_motion(self) -> None:
        flat = np.full((SIZE, SIZE), 0.5)
        self.assertLess(float(np.abs(lucas_kanade(flat, flat)).max()), 1e-6)


class ThresholdTest(unittest.TestCase):
    def test_jitter_below_the_threshold_is_no_motion(self) -> None:
        np.testing.assert_array_equal(threshold_displacement(np.array([[0.2, 0.1]]), 0.5), [[0.0, 0.0]])

    def test_the_threshold_comes_off_the_length_and_the_direction_stays(self) -> None:
        np.testing.assert_allclose(threshold_displacement(np.array([[3.0, 4.0]]), 1.0), [[2.4, 3.2]])

    def test_no_threshold_leaves_it(self) -> None:
        np.testing.assert_allclose(threshold_displacement(np.array([[0.3, -0.2]]), 0.0), [[0.3, -0.2]])


class DisplacementVelocityTest(unittest.TestCase):
    def test_a_width_per_second_is_one(self) -> None:
        velocity = displacement_velocity(np.array([0.1, 0.0]), 0.1, 200, 100, (1.0, 1.0))
        np.testing.assert_allclose(velocity, [1.0, 0.0])

    def test_vertical_motion_is_counted_in_widths(self) -> None:
        # 10 pixels down in a 200x100 image is 0.1 of the height, 0.05 of the width
        velocity = displacement_velocity(np.array([0.0, 0.1]), 1.0, 200, 100, (1.0, 1.0))
        np.testing.assert_allclose(velocity, [0.0, 0.05])

    def test_strength_multiplies_each_axis(self) -> None:
        np.testing.assert_allclose(displacement_velocity(np.array([0.1, 0.1]), 1.0, 10, 10, (3.0, -1.0)), [0.3, -0.1])


class ShapeTest(unittest.TestCase):
    def test_the_trail_bridge_sees_what_the_old_shaping_gave(self) -> None:
        # The shaping took its threshold after scale and strength: max(|d| * scale - 0.1, 0) at scale 0.2.
        # Taken off the displacement first as 0.1 / 0.2 = 0.5 px, then scaled, the trail bridge's velocity is the same.
        displacement = np.array([[0.3, 0.0], [1.0, 0.0], [2.0, 1.0], [6.0, -8.0]])
        old_length = np.maximum(np.linalg.norm(displacement * 0.2, axis=-1, keepdims=True) - 0.1, 0.0)
        old = np.clip(np.divide(displacement, np.linalg.norm(displacement, axis=-1, keepdims=True)) * old_length, -1.0, 1.0)
        new = shape_velocity(threshold_displacement(displacement, 0.5) * 0.2, 1.0)
        np.testing.assert_allclose(new, old, atol=1e-9)

    def test_the_velocity_is_clamped(self) -> None:
        self.assertEqual(float(shape_velocity(np.array([[10.0, 0.0]]), 1.0)[0, 0]), 1.0)

    def test_boost_lifts_small_motions(self) -> None:
        np.testing.assert_allclose(shape_velocity(np.array([[0.25, 0.0]]), 0.5), [[0.5, 0.0]])


if __name__ == '__main__':
    unittest.main()
