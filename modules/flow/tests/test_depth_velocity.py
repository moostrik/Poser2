"""The depth velocity's numpy reference: z over a window of depth samples, along the motion, averaging
in only the neighbours within the depth tolerance and only where the window's coverage is enough, in
view widths per frame; and the noise floor taken off it."""

import unittest

import numpy as np

from modules.flow import depth_threshold, window_depth_displacement

SIZE: int = 24
SPAN: float = 0.5       # metres a view width spans per metre of depth
TOLERANCE: float = 0.1
STEP: tuple[float, float] = (1.0, 1.0)


def still() -> np.ndarray:
    return np.zeros((SIZE, SIZE, 2))


def uniform(depth: float) -> np.ndarray:
    return np.full((SIZE, SIZE), depth)


def interior(field: np.ndarray, margin: int = 8) -> np.ndarray:
    return field[margin:-margin, margin:-margin]


def z(current: np.ndarray, previous: np.ndarray, moved: np.ndarray | None = None, window: int = 2,
      step: tuple[float, float] = STEP, tolerance: float = TOLERANCE, coverage: float = 0.0, span: float = SPAN) -> np.ndarray:
    return window_depth_displacement(current, previous, still() if moved is None else moved, window, step, tolerance, coverage, span)


class DepthDisplacementTest(unittest.TestCase):
    def test_coming_closer_is_positive(self) -> None:
        # 10 cm closer at 1.9 m, where a view width of span 0.5 is 0.95 m
        np.testing.assert_allclose(interior(z(uniform(1.9), uniform(2.0), tolerance=0.5)), 0.1 / 0.95)

    def test_going_away_is_negative(self) -> None:
        np.testing.assert_allclose(interior(z(uniform(2.1), uniform(2.0), tolerance=0.5)), -0.1 / 1.05)

    def test_the_same_step_further_away_is_smaller_like_a_sideways_one(self) -> None:
        near = z(uniform(1.99), uniform(2.0))
        far = z(uniform(3.99), uniform(4.0))
        np.testing.assert_allclose(interior(far) / interior(near), 1.99 / 3.99)

    def test_something_passing_over_the_pixels_is_no_motion(self) -> None:
        self.assertEqual(float(np.abs(z(uniform(1.2), uniform(3.0))).max()), 0.0)

    def test_no_depth_now_or_before_is_no_motion(self) -> None:
        self.assertEqual(float(np.abs(z(uniform(0.0), uniform(2.0))).max()), 0.0)
        self.assertEqual(float(np.abs(z(uniform(2.0), uniform(0.0))).max()), 0.0)

    def test_no_view_span_is_no_motion(self) -> None:
        self.assertEqual(float(np.abs(z(uniform(1.9), uniform(2.0), tolerance=0.5, span=0.0)).max()), 0.0)

    def test_the_background_beside_the_player_is_not_averaged_in(self) -> None:
        # the player on the left, 1 cm closer; the wall on the right, 5 cm further, both within the tolerance in time
        current, previous = uniform(1.0), uniform(1.01)
        current[:, SIZE // 2:], previous[:, SIZE // 2:] = 3.0, 2.95
        field = z(current, previous)
        self.assertAlmostEqual(float(field[SIZE // 2, SIZE // 2 - 1]), 0.01 / (1.0 * SPAN))       # beside the wall: only the player counts
        self.assertAlmostEqual(float(field[SIZE // 2, SIZE // 2]), -0.05 / (3.0 * SPAN))           # and the other way round

    def test_a_pixel_with_too_little_of_its_window_averaged_in_reads_no_motion(self) -> None:
        # one stray reading between the player and the wall: only it is within the tolerance of itself
        current, previous = uniform(1.0), uniform(1.01)
        current[SIZE // 2, SIZE // 2], previous[SIZE // 2, SIZE // 2] = 1.5, 1.45
        field = z(current, previous, coverage=0.5)
        self.assertEqual(float(field[SIZE // 2, SIZE // 2]), 0.0)
        self.assertAlmostEqual(float(field[SIZE // 2, SIZE // 2 + 3]), 0.01 / (1.0 * SPAN))       # the player beside it still reads

    def test_a_silhouette_reads_where_enough_of_the_window_is_the_player(self) -> None:
        current, previous = uniform(1.0), uniform(1.01)
        current[:, SIZE // 2:], previous[:, SIZE // 2:] = 3.0, 3.0
        field = z(current, previous, coverage=0.7)                                                  # 18 of 25 needed
        self.assertEqual(float(field[SIZE // 2, SIZE // 2 - 1]), 0.0)                              # beside the wall, 3 of 5 columns are the player: 15 of 25
        self.assertAlmostEqual(float(field[SIZE // 2, SIZE // 2 - 2]), 0.01 / (1.0 * SPAN))       # one further, 4 of 5: 20 of 25

    def test_independent_noise_drops_with_the_root_of_the_samples(self) -> None:
        # depth samples three flow pixels apart, as a coarse depth stretched onto the view
        rng = np.random.default_rng(0)
        samples: int = 48
        current = np.repeat(np.repeat(2.0 + rng.normal(0.0, 0.03, (samples, samples)), 3, axis=0), 3, axis=1)
        previous = np.repeat(np.repeat(2.0 + rng.normal(0.0, 0.03, (samples, samples)), 3, axis=0), 3, axis=1)
        moved = np.zeros(current.shape + (2,))
        one = z(current, previous, moved, window=0, step=(3.0, 3.0), tolerance=1.0)
        many = z(current, previous, moved, window=2, step=(3.0, 3.0), tolerance=1.0)
        ratio: float = float(np.std(many[12:-12, 12:-12]) / np.std(one[12:-12, 12:-12]))
        self.assertAlmostEqual(ratio, 1.0 / 5.0, delta=0.05)       # 25 samples

    def test_a_surface_sliding_sideways_over_a_slope_reads_no_motion_along_its_motion(self) -> None:
        rows, columns = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
        previous = 2.0 + 0.01 * columns                 # a slope: 1 cm per pixel
        current = 2.0 + 0.01 * (columns - 3.0)          # slid 3 pixels to the right
        moved = still()
        moved[..., 0] = 3.0
        self.assertGreater(float(np.abs(interior(z(current, previous))).min()), 0.02)     # 3 cm per frame read as about 0.026 view widths
        self.assertLess(float(np.abs(interior(z(current, previous, moved))).max()), 1e-9)

    def test_a_surface_moving_toward_the_camera_along_its_motion_still_reads_so(self) -> None:
        moved = still()
        moved[..., 0] = 2.0
        np.testing.assert_allclose(interior(z(uniform(1.99), uniform(2.0), moved)), 0.01 / (1.99 * SPAN))


class DepthThresholdTest(unittest.TestCase):
    def test_the_threshold_clears_the_floor_and_stays_continuous(self) -> None:
        np.testing.assert_allclose(depth_threshold(np.array([0.05, -0.05, 0.3, -0.3]), 0.1), [0.0, 0.0, 0.2, -0.2])


if __name__ == '__main__':
    unittest.main()
