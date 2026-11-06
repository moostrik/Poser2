"""The 3D bridge's trail arithmetic: the Ema weight, a trail's direction and its advection."""

import math
import unittest

import numpy as np

from modules.flow import advect, decay_weight, direction


class DecayWeightTest(unittest.TestCase):
    def test_one_time_constant_leaves_one_over_e(self) -> None:
        self.assertAlmostEqual(decay_weight(0.025, 25.0), 1 / math.e)

    def test_steps_compose_whatever_the_frame_rate(self) -> None:
        self.assertAlmostEqual(decay_weight(1 / 60, 150.0) ** 2, decay_weight(1 / 30, 150.0))

    def test_no_time_constant_forgets_at_once(self) -> None:
        self.assertEqual(decay_weight(1 / 60, 0.0), 0.0)


class DirectionTest(unittest.TestCase):
    def test_the_fast_trail_gives_the_direction(self) -> None:
        np.testing.assert_allclose(direction(np.array([[3.0, 4.0, 0.0]]), np.array([[1.0, 0.0, 0.0]])), [[0.6, 0.8, 0.0]])

    def test_a_stopped_fast_trail_takes_the_slow_ones(self) -> None:
        np.testing.assert_allclose(direction(np.zeros((1, 3)), np.array([[0.0, 0.0, 2.0]])), [[0.0, 0.0, 1.0]])

    def test_no_motion_has_no_direction(self) -> None:
        np.testing.assert_allclose(direction(np.zeros((1, 3)), np.zeros((1, 3))), np.zeros((1, 3)))


class AdvectTest(unittest.TestCase):
    def test_no_amount_leaves_the_field(self) -> None:
        field = np.random.default_rng(0).random((8, 8, 2))
        np.testing.assert_allclose(advect(field, np.full((8, 8, 2), 0.25), 0.0), field)

    def test_a_field_moves_with_its_displacement(self) -> None:
        field = np.zeros((1, 8, 1))
        field[0, 2] = 1.0
        moved = advect(field, np.tile([2 / 8, 0.0], (1, 8, 1)), 1.0)       # two pixels right
        self.assertEqual(int(np.argmax(moved[0, :, 0])), 4)


if __name__ == '__main__':
    unittest.main()
