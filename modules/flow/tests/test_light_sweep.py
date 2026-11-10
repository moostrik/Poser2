"""The sun's sweep: the axis the light mostly travels along, its direction, the slide across, the lines that
cover the cells it enters from a side, and the map from a place to the sheared texel of the lines around it."""

import math
import unittest

import numpy as np

from modules.flow.fluid_3d.shaders.light_sweep_3d import sweep_lines, sweep_map, sweep_size


class TestSweepLines(unittest.TestCase):

    def test_sun_straight_above_sweeps_down_the_rows(self) -> None:
        axis, direction, slope, start, count, step = sweep_lines((72, 128, 16), (0.0, -1.0, 0.0))
        self.assertEqual((axis, direction), (1, 1))         # y down the screen: from row 0 to higher rows
        self.assertEqual(slope, (0.0, 0.0))
        self.assertEqual(start, (-1, -1))                   # one line before the first cell and one past the last
        self.assertEqual(count, (74, 18))
        self.assertAlmostEqual(step, 1.0)

    def test_tilted_sun_slides_across_and_steps_longer(self) -> None:
        toward = (0.6, -0.8, 0.0)
        axis, direction, slope, start, count, step = sweep_lines((72, 128, 16), toward)
        self.assertEqual((axis, direction), (1, 1))
        self.assertAlmostEqual(slope[0], -0.75)             # the light travels toward lower x
        self.assertAlmostEqual(slope[1], 0.0)
        self.assertEqual(start[0], -1)                      # it slides in from the right: lines past the last cell
        self.assertEqual(count[0], 72 + math.ceil(0.75 * 128) + 2)
        self.assertAlmostEqual(step, 1.25)

    def test_sun_from_the_front_sweeps_into_the_depth(self) -> None:
        axis, direction, _, _, _, _ = sweep_lines((72, 128, 16), (0.1, 0.1, -0.99))
        self.assertEqual((axis, direction), (2, 1))         # z away from the viewer: front layer first


class TestSweepMap(unittest.TestCase):

    def test_a_cell_maps_to_the_line_through_it(self) -> None:
        size = (72, 128, 16)
        for toward in ((0.6, -0.8, 0.0), (-0.3, -0.9, 0.3), (0.2, 0.95, -0.2)):
            toward = tuple(c / math.sqrt(sum(v * v for v in toward)) for c in toward)
            axis, direction, slope, start, count, _ = sweep_lines(size, toward)
            sheared = sweep_size(size, toward)
            mapping = np.array(sweep_map(size, toward)).reshape(4, 4)
            others = [a for a in range(3) if a != axis]
            for line, k in (((5, 3), 0), ((40, 7), 37), ((-2, 1), 90)):
                along = k if direction > 0 else size[axis] - 1 - k
                place = np.zeros(3)
                place[axis] = (along + 0.5) / size[axis]
                for b, i, s in zip(others, line, slope):
                    place[b] = (i + 0.5 + s * k) / size[b]      # where line i passes at layer k
                texel = (mapping @ np.append(place, 1.0))[:3] * np.array(sheared)
                expected = np.zeros(3)
                expected[axis] = along + 0.5
                for b, i, first in zip(others, line, start):
                    expected[b] = i - first + 0.5               # its column's middle
                np.testing.assert_allclose(texel, expected, atol=1e-9)
            self.assertEqual(sheared[others[0]], count[0])


if __name__ == '__main__':
    unittest.main()
