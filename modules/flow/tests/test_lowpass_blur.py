"""The lowpass's blur values: its weights, its dispatches and its sigma as diffusion."""

import math
import unittest

from modules.flow import blur_dispatches, blur_sigma, gaussian_weights


class TestBlur(unittest.TestCase):

    def test_the_weights_are_normalized_over_both_sides(self) -> None:
        w = gaussian_weights(1.7)
        self.assertAlmostEqual(float(w[0] + 2.0 * w[1:].sum()), 1.0)

    def test_k_blurs_of_sigma_over_root_k_add_up_to_one_of_sigma_within_the_radius(self) -> None:
        count, each = blur_dispatches(6.0, 8)
        self.assertGreater(count, 1)
        self.assertAlmostEqual(count * each * each, 6.0 * 6.0)       # variances add
        self.assertLessEqual(3.0 * each, 8)

    def test_the_spread_in_a_second_is_the_same_at_any_frame_rate(self) -> None:
        width = 448
        for fps in (25.0, 60.0, 144.0):
            sigma_step = blur_sigma(0.081, 1.0 / fps, width)
            sigma_second = math.sqrt(fps) * sigma_step
            self.assertAlmostEqual(sigma_second * 2.0 * math.sqrt(2.0 * math.log(2.0)) / width, 0.081)


if __name__ == '__main__':
    unittest.main()
