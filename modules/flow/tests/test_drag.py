"""The 3D fluid's drag share: the step's share of the gap to the body's velocity, whatever the frame rate."""

import math
import unittest

from modules.flow import drag_share


class DragShareTest(unittest.TestCase):
    def test_one_drag_time_closes_all_but_one_over_e(self) -> None:
        self.assertAlmostEqual(drag_share(0.05, 50.0), 1.0 - 1.0 / math.e)

    def test_steps_compose_whatever_the_frame_rate(self) -> None:
        remaining_60 = (1.0 - drag_share(1 / 60, 50.0)) ** 2
        self.assertAlmostEqual(remaining_60, 1.0 - drag_share(1 / 30, 50.0))

    def test_no_drag_time_sets_the_velocity_at_once(self) -> None:
        self.assertEqual(drag_share(1 / 60, 0.0), 1.0)


if __name__ == '__main__':
    unittest.main()
