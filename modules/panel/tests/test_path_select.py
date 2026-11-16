"""The path select's levels: a dotted path list split a level at a time, steps and textures side by side in the
order the paths come, and the step a path lies in."""

import unittest

from ..controls import path_level, path_step

PATHS: list[str] = ['input.image', 'input.depth', 'depth.depth', 'depth_mask.depth_mask', 'depth_mask.mask',
                    'velocity', 'xy.solve.level', 'output.image']


class PathSelectTest(unittest.TestCase):
    def test_the_top_level_lists_steps_and_textures_in_order(self) -> None:
        self.assertEqual(path_level(PATHS, ''), {'input.': 'input ›', 'depth.': 'depth ›', 'depth_mask.': 'depth_mask ›',
                                                 'velocity': 'velocity', 'xy.': 'xy ›', 'output.': 'output ›'})

    def test_a_step_lists_what_lies_one_level_below_it(self) -> None:
        self.assertEqual(path_level(PATHS, 'depth_mask.'), {'depth_mask.depth_mask': 'depth_mask', 'depth_mask.mask': 'mask'})
        self.assertEqual(path_level(PATHS, 'xy.'), {'xy.solve.': 'solve ›'})
        self.assertEqual(path_level(PATHS, 'xy.solve.'), {'xy.solve.level': 'level'})

    def test_a_path_lies_in_the_step_before_its_last_dot(self) -> None:
        self.assertEqual(path_step('depth_mask.mask'), 'depth_mask.')
        self.assertEqual(path_step('xy.solve.level'), 'xy.solve.')
        self.assertEqual(path_step('velocity'), '')


if __name__ == '__main__':
    unittest.main()
