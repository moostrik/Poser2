"""Scene flow's numpy references: Lucas–Kanade finds how far an image moved, and depth followed along
that displacement reads no motion toward the camera for a surface that only slides sideways."""

import unittest

import numpy as np

from modules.flow import depth_speed, lucas_kanade, shape_displacement, threshold_displacement, warped

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


class ShapeTest(unittest.TestCase):
    def test_the_trail_bridge_sees_what_the_old_shaping_gave(self) -> None:
        # The shaping took its threshold after scale and strength: max(|d| * scale - 0.1, 0) at scale 0.2.
        # Taken off the displacement first as 0.1 / 0.2 = 0.5 px, the trail bridge's velocity is the same.
        displacement = np.array([[0.3, 0.0], [1.0, 0.0], [2.0, 1.0], [6.0, -8.0]])
        old_length = np.maximum(np.linalg.norm(displacement * 0.2, axis=-1, keepdims=True) - 0.1, 0.0)
        old = np.clip(np.divide(displacement, np.linalg.norm(displacement, axis=-1, keepdims=True)) * old_length, -1.0, 1.0)
        new = shape_displacement(threshold_displacement(displacement, 0.5), 0.2, (1.0, 1.0), 1.0)
        np.testing.assert_allclose(new, old, atol=1e-9)

    def test_the_velocity_is_clamped(self) -> None:
        self.assertEqual(float(shape_displacement(np.array([[50.0, 0.0]]), 0.2, (1.0, 1.0), 1.0)[0, 0]), 1.0)


class WarpedDepthTest(unittest.TestCase):
    def test_a_surface_sliding_sideways_over_a_slope_reads_no_motion_toward_the_camera_along_its_motion(self) -> None:
        rows, columns = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
        previous = 2.0 + 0.01 * columns                 # a slope: 1 cm per pixel
        current = 2.0 + 0.01 * (columns - 3.0)          # slid 3 pixels to the right
        displacement = np.zeros((SIZE, SIZE, 2))
        displacement[..., 0] = 3.0
        at_pixel = interior(depth_speed(current, previous, 1.0 / 60.0, 0.1))
        along = interior(depth_speed(current, warped(previous, displacement), 1.0 / 60.0, 0.1))
        self.assertGreater(float(np.abs(at_pixel).min()), 1.0)      # 3 cm per frame read as 1.8 m/s
        self.assertLess(float(np.abs(along).max()), 1e-9)

    def test_a_surface_moving_toward_the_camera_still_reads_so(self) -> None:
        previous = np.full((SIZE, SIZE), 2.0)
        current = np.full((SIZE, SIZE), 1.99)           # 1 cm closer
        displacement = np.zeros((SIZE, SIZE, 2))
        speed = depth_speed(current, warped(previous, displacement), 1.0 / 60.0, 0.1)
        self.assertAlmostEqual(float(speed.mean()), 0.6, places=6)


if __name__ == '__main__':
    unittest.main()
