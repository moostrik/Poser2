"""The expansion: its mean over the water as the GPU sums it, the divergence the solve makes, the presence capped."""

import unittest

import numpy as np

from modules.flow import divergence_target, expansion_mean, weighted_sum


class TestExpansionMean(unittest.TestCase):

    def test_the_sum_in_workgroups_and_lanes_is_the_mean(self) -> None:
        expansion = np.random.default_rng(3).random((12, 40, 24))     # not a multiple of the group on y, z
        self.assertAlmostEqual(expansion_mean(expansion), float(expansion.mean()))

    def test_an_obstacles_cells_count_for_nothing(self) -> None:
        expansion = np.ones((4, 8, 8))
        obstacle = np.zeros_like(expansion)
        obstacle[:, :, :4] = 1.0
        expansion[:, :, :4] = 5.0
        self.assertAlmostEqual(expansion_mean(expansion, obstacle), 1.0)


class TestDivergenceTarget(unittest.TestCase):

    def setUp(self) -> None:
        self.expansion = np.zeros((4, 8, 8))
        self.expansion[1:3, 2:6, 2:6] = 1.0     # the body, fully there

    def test_in_a_closed_box_the_target_sums_to_zero(self) -> None:
        phi = divergence_target(self.expansion, 0.5, closed=True)
        self.assertAlmostEqual(float(phi.sum()), 0.0, places=9)

    def test_a_push_in_one_place_is_a_slight_pull_everywhere_else(self) -> None:
        phi = divergence_target(self.expansion, 0.5, closed=True)
        self.assertGreater(float(phi[1, 3, 3]), 0.0)
        self.assertLess(float(phi[0, 0, 0]), 0.0)
        self.assertAlmostEqual(float(phi[0, 0, 0]), -0.5 * self.expansion.mean())

    def test_through_an_open_wall_the_target_is_the_rate_times_the_presence(self) -> None:
        np.testing.assert_allclose(divergence_target(self.expansion, -0.5, closed=False), -0.5 * self.expansion)


class TestPresence(unittest.TestCase):

    def test_the_expansions_presence_is_capped_at_one(self) -> None:
        amount = weighted_sum([np.array([[3.0, 4.0, 0.0]]), np.array([[1.0]])], [1.0, 1.0], lengths=True, capped=True)
        self.assertAlmostEqual(float(amount[0, 0]), 1.0)


if __name__ == '__main__':
    unittest.main()
