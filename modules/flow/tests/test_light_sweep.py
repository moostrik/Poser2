"""The sun's sweep: the axis the light mostly travels along, its direction, the slide across and the margin
that covers the cells it enters from a side."""

import math
import unittest

from modules.flow.fluid_3d.shaders.light_sweep_3d import sweep_lines


class TestSweepLines(unittest.TestCase):

    def test_sun_straight_above_sweeps_down_the_rows(self) -> None:
        axis, direction, slope, margin, step = sweep_lines((72, 128, 16), (0.0, -1.0, 0.0))
        self.assertEqual((axis, direction), (1, 1))         # y down the screen: from row 0 to higher rows
        self.assertEqual(slope, (0.0, 0.0))
        self.assertEqual(margin, (1, 1))
        self.assertAlmostEqual(step, 1.0)

    def test_tilted_sun_slides_across_and_steps_longer(self) -> None:
        toward = (0.6, -0.8, 0.0)
        axis, direction, slope, margin, step = sweep_lines((72, 128, 16), toward)
        self.assertEqual((axis, direction), (1, 1))
        self.assertAlmostEqual(slope[0], -0.75)             # the light travels toward lower x
        self.assertAlmostEqual(slope[1], 0.0)
        self.assertEqual(margin[0], math.ceil(0.75 * 128) + 1)
        self.assertAlmostEqual(step, 1.25)

    def test_sun_from_the_front_sweeps_into_the_depth(self) -> None:
        axis, direction, _, _, _ = sweep_lines((72, 128, 16), (0.1, 0.1, -0.99))
        self.assertEqual((axis, direction), (2, 1))         # z away from the viewer: front layer first


if __name__ == '__main__':
    unittest.main()
