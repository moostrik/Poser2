"""The depth velocity's helpers: the alpha–beta weights, and the numpy references of the step filter and of
the noise floor taken off z."""

import unittest

import numpy as np

from modules.flow import depth_threshold, filter_depth_steps, track_weights

SIZE: int = 24
TOLERANCE: float = 0.1
STEP: tuple[float, float] = (1.0, 1.0)
DT: float = 1.0 / 30.0


def uniform(depth: float) -> np.ndarray:
    return np.full((SIZE, SIZE), depth)


class WeightsTest(unittest.TestCase):
    def test_no_time_takes_the_measurement_whole(self) -> None:
        self.assertEqual(track_weights(DT, 0.0, 1.0), (1.0, 1.0))

    def test_one_time_constant_leaves_one_over_e(self) -> None:
        alpha, beta = track_weights(0.1, 100.0, 1.0)
        self.assertAlmostEqual(float(alpha), 1.0 - np.exp(-1.0))
        self.assertAlmostEqual(float(beta), float(alpha) ** 2 / (2.0 - float(alpha)))

    def test_the_share_of_the_window_scales_the_correction(self) -> None:
        self.assertAlmostEqual(float(track_weights(DT, 0.0, 0.5)[0]), 0.5)


class StepFilterTest(unittest.TestCase):
    """Before the track, a sample whose 3×3 depth samples span two surfaces has no depth."""

    def test_a_step_is_an_edge_by_the_edge_value(self) -> None:
        # a hand 15 cm before the chest: an edge at edge 0.1, the same surface at 0.2
        rows, columns = np.mgrid[0:SIZE, 0:SIZE]
        depth = np.where(columns < SIZE // 2, 0.85, 1.0)
        self.assertEqual(float(filter_depth_steps(depth, STEP, 0.1)[5, SIZE // 2]), 0.0)
        self.assertEqual(float(filter_depth_steps(depth, STEP, 0.2)[5, SIZE // 2]), 1.0)

    def test_a_flat_or_sloping_surface_is_unchanged(self) -> None:
        rows, columns = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
        flat, slope = uniform(1.0), 1.0 + 0.02 * columns               # 2 cm per sample, under the tolerance
        np.testing.assert_array_equal(filter_depth_steps(flat, STEP, TOLERANCE), flat)
        np.testing.assert_array_equal(filter_depth_steps(slope, STEP, TOLERANCE), slope)

    def test_a_silhouette_loses_a_band_a_sample_wide_on_each_side(self) -> None:
        depth = uniform(2.0)
        depth[:, :SIZE // 2] = 0.7                                       # the player on the left, the wall on the right
        kept = filter_depth_steps(depth, STEP, TOLERANCE)
        self.assertTrue(np.all(kept[:, SIZE // 2 - 1:SIZE // 2 + 1] == 0.0))
        np.testing.assert_array_equal(kept[:, :SIZE // 2 - 1], 0.7)
        np.testing.assert_array_equal(kept[:, SIZE // 2 + 1:], 2.0)

    def test_the_band_is_one_depth_sample_wide_at_its_spacing(self) -> None:
        depth = uniform(2.0)
        depth[:, :12] = 0.7
        kept = filter_depth_steps(depth, (3.0, 3.0), TOLERANCE)        # depth samples three pixels apart
        self.assertTrue(np.all(kept[:, 9:15] == 0.0))
        self.assertTrue(np.all(kept[:, :9] == 0.7) and np.all(kept[:, 15:] == 2.0))

    def test_no_depth_beside_it_is_no_step(self) -> None:
        depth = uniform(1.0)
        depth[:, :SIZE // 2] = 0.0
        np.testing.assert_array_equal(filter_depth_steps(depth, STEP, TOLERANCE), depth)

class DepthThresholdTest(unittest.TestCase):
    def test_the_threshold_clears_the_floor_and_stays_continuous(self) -> None:
        np.testing.assert_allclose(depth_threshold(np.array([0.05, -0.05, 0.3, -0.3]), 0.1), [0.0, 0.0, 0.2, -0.2])


if __name__ == '__main__':
    unittest.main()
