"""The z position's references: the motion's, a trail's step, an output's mix; and the lowpass's blur as diffusion."""

import math
import unittest

import numpy as np

from modules.flow import blur_dispatches, blur_sigma, gaussian_blur, gaussian_weights, z_position, z_position_mix, z_position_trail


def place(z: np.ndarray) -> np.ndarray:
    """A z position, red over green."""
    return z[..., 0] / z[..., 1]


class TestMotionZPosition(unittest.TestCase):

    def test_the_range_maps_near_to_far_on_zero_to_one(self) -> None:
        z = z_position(1.0 / np.array([0.1, 0.9, 1.7]), 0.1, 1.7)
        np.testing.assert_allclose(place(z), [0.0, 0.5, 1.0], atol=1e-9)
        np.testing.assert_allclose(z[..., 1], 1.0)

    def test_without_depth_is_no_z_position_not_the_front(self) -> None:
        z = z_position(np.array([0.0]), 0.1, 1.7)
        np.testing.assert_array_equal(z, 0.0)

    def test_a_track_out_of_the_range_is_held_to_its_ends(self) -> None:
        z = z_position(np.array([1.0 / 0.05, 1.0 / 3.0]), 0.1, 1.7)
        np.testing.assert_allclose(place(z), [0.0, 1.0], atol=1e-9)

    def test_the_depth_mask_says_where_the_depth_is_the_players(self) -> None:
        z = z_position(np.array([1.0 / 0.9, 1.0 / 0.9]), 0.1, 1.7, depth_mask=np.array([1.0, 0.0]))
        np.testing.assert_allclose(z[..., 1], [1.0, 0.0])

    def test_the_weight_is_the_depth_mask_times_the_mask(self) -> None:
        z = z_position(np.array([1.0 / 0.9]), 0.1, 1.7, np.array([0.25]), np.array([0.5]))
        self.assertAlmostEqual(float(z[0, 1]), 0.125)
        self.assertAlmostEqual(float(place(z)[0]), 0.5)


class TestTrailZPosition(unittest.TestCase):

    def test_a_trail_keeps_the_depth_of_the_motion_it_holds_when_a_still_body_takes_its_place(self) -> None:
        arm, torso = np.array([[0.2, 1.0]]), np.array([[0.7, 1.0]])
        held = np.zeros((1, 2))
        trail = np.zeros((1, 3))
        for _ in range(20):         # the arm moves at the pixel
            velocity = np.array([[1.0, 0.0, 0.0]])
            held = z_position_trail(held, trail, velocity, arm, 0.0, 0.9)
            trail = 0.0 * trail + velocity
        for _ in range(20):         # it left; the still torso is there now
            held = z_position_trail(held, trail, np.zeros((1, 3)), torso, 0.9, 0.9)
            trail = 0.9 * trail
        self.assertAlmostEqual(float(place(held)[0]), 0.2, places=6)

    def test_two_motions_average_by_the_speed_each_brings(self) -> None:
        held = z_position_trail(np.zeros((1, 2)), np.zeros((1, 3)), np.array([[3.0, 0.0, 0.0]]), np.array([[0.0, 1.0]]), 0.5, 0.5)
        held = z_position_trail(held, np.array([[1.5, 0.0, 0.0]]), np.array([[1.0, 0.0, 0.0]]), np.array([[1.0, 1.0]]), 0.5, 0.5)
        # the first at 3 × 0.5 × 0.5 (one step older), the second at 1 × 0.5: place 1 by 0.5 of 1.25
        self.assertAlmostEqual(float(place(held)[0]), 0.5 / 1.25)


class TestZPositionMix(unittest.TestCase):

    def setUp(self) -> None:
        self.trails = (np.array([[1.0, 0.0, 0.0]]), np.array([[2.0, 0.0, 0.0]]), np.array([[1.0, 0.0, 0.0]]))
        self.z = (np.array([[0.2, 1.0]]), np.array([[2 * 0.5, 2.0]]), np.array([[0.9, 1.0]]))    # places 0.2, 0.5, 0.9

    def mix(self, shares: list[float], readings: list[np.ndarray] | None = None) -> float:
        readings = readings or [np.array([[1.0, 0.0, 0.0]])] * 6
        return float(place(z_position_mix(readings, shares, self.trails, self.z))[0])

    def test_each_reading_reads_its_trails_place(self) -> None:
        self.assertAlmostEqual(self.mix([1, 0, 0, 0, 0, 0]), 0.2)
        self.assertAlmostEqual(self.mix([0, 0, 1, 1, 0, 0]), 0.5)
        self.assertAlmostEqual(self.mix([0, 0, 0, 0, 1, 1]), 0.9)

    def test_the_change_reads_the_faster_trails_place(self) -> None:
        self.assertAlmostEqual(self.mix([0, 1, 0, 0, 0, 0]), 0.5)

    def test_the_readings_weigh_by_their_sizes_at_their_shares(self) -> None:
        readings = [np.array([[3.0, 0.0, 0.0]])] + [np.array([[1.0, 0.0, 0.0]])] * 5
        self.assertAlmostEqual(self.mix([1, 0, 0, 0, 1, 0], readings), (3 * 0.2 + 0.9) / 4)

    def test_motion_without_depth_does_not_pull_it(self) -> None:
        self.z = (np.array([[0.0, 0.0]]), *self.z[1:])
        self.assertAlmostEqual(self.mix([1, 0, 0, 0, 1, 0]), 0.9)


class TestBlur(unittest.TestCase):

    def test_the_weights_are_normalized_over_both_sides(self) -> None:
        w = gaussian_weights(1.7)
        self.assertAlmostEqual(float(w[0] + 2.0 * w[1:].sum()), 1.0)

    def test_k_blurs_of_sigma_over_root_k_are_one_of_sigma(self) -> None:
        rng = np.random.default_rng(1)
        field = np.zeros((81, 81, 1))
        field[40, 40] = 1.0
        count, each = blur_dispatches(6.0, 8)
        self.assertGreater(count, 1)
        once = gaussian_blur(field, 6.0)
        split = field
        for _ in range(count):
            split = gaussian_blur(split, each)
        np.testing.assert_allclose(split, once, atol=2e-4)
        self.assertIsNotNone(rng)

    def test_the_spread_in_a_second_is_the_same_at_any_frame_rate(self) -> None:
        width = 448
        for fps in (25.0, 60.0, 144.0):
            sigma_step = blur_sigma(0.081, 1.0 / fps, width)
            sigma_second = math.sqrt(fps) * sigma_step
            self.assertAlmostEqual(sigma_second * 2.0 * math.sqrt(2.0 * math.log(2.0)) / width, 0.081)


if __name__ == '__main__':
    unittest.main()
