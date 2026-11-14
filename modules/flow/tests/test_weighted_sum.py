"""The bridge's mix: an amount from the readings' sizes and the depth mask."""

import unittest

import numpy as np

from modules.flow import weighted_sum


class TestAmount(unittest.TestCase):

    def test_a_still_body_is_there_by_the_depth_mask_at_its_share(self) -> None:
        still = [np.zeros((1, 3))] * 6
        amount = weighted_sum([*still, np.array([[0.8]])], [1, 0, 0, 0, 0, 0, 0.5], lengths=True)
        self.assertAlmostEqual(float(amount[0, 0]), 0.5 * 0.8)

    def test_the_depth_mask_adds_to_the_readings_speeds(self) -> None:
        readings = [np.array([[0.3, 0.4, 0.0]])] * 6
        amount = weighted_sum([*readings, np.array([[1.0]])], [1, 0, 0, 0, 0, 0, 0.25], lengths=True)
        self.assertAlmostEqual(float(amount[0, 0]), 0.5 + 0.25)

    def test_a_share_of_zero_adds_nothing(self) -> None:
        amount = weighted_sum([np.array([[0.3, 0.4, 0.0]]), np.array([[1.0]])], [0.0, 0.0], lengths=True)
        np.testing.assert_allclose(amount, 0.0)


if __name__ == '__main__':
    unittest.main()
