"""The 3D bridge's trail weight: the Ema weight."""

import math
import unittest

from modules.flow import decay_weight


class DecayWeightTest(unittest.TestCase):
    def test_one_time_constant_leaves_one_over_e(self) -> None:
        self.assertAlmostEqual(decay_weight(0.025, 25.0), 1 / math.e)

    def test_steps_compose_whatever_the_frame_rate(self) -> None:
        self.assertAlmostEqual(decay_weight(1 / 60, 150.0) ** 2, decay_weight(1 / 30, 150.0))

    def test_no_time_constant_forgets_at_once(self) -> None:
        self.assertEqual(decay_weight(1 / 60, 0.0), 0.0)


if __name__ == '__main__':
    unittest.main()
