"""The orbit camera frames the box's center, turns with the drag, keeps its pitch and zoom in their
limits, and knows which side of the box it is on."""

import math
import unittest

import numpy as np

from modules.flow import Orbit, box_size


def clip(orbit: Orbit, point: tuple[float, float, float], box: tuple[float, float, float]) -> np.ndarray:
    matrix = orbit.view_projection(box, 9 / 16)
    p = matrix @ np.array([*point, 1.0])
    return p[:3] / p[3]


class OrbitTest(unittest.TestCase):
    BOX = box_size(448, 800, 0.14)

    def test_the_center_is_in_the_middle_from_any_angle(self) -> None:
        orbit = Orbit()
        for dx, dy in ((0.0, 0.0), (0.3, 0.1), (-0.7, -0.4)):
            orbit.turn(dx, dy)
            np.testing.assert_allclose(clip(orbit, (0.0, 0.0, 0.0), self.BOX)[:2], (0.0, 0.0), atol=1e-9)

    def test_the_front_view_frames_the_box_inside_the_picture(self) -> None:
        corner = clip(Orbit(), (0.5, self.BOX[1] / 2, self.BOX[2] / 2), self.BOX)
        self.assertTrue(all(abs(c) <= 1.0 for c in corner[:2]))

    def test_a_drag_right_turns_the_box_right(self) -> None:
        orbit = Orbit()
        orbit.turn(0.1, 0.0)
        front = clip(orbit, (0.0, 0.0, self.BOX[2] / 2), self.BOX)     # the front face's center
        self.assertGreater(front[0], 0.0)

    def test_pitch_and_zoom_stay_in_their_limits(self) -> None:
        orbit = Orbit()
        orbit.turn(0.0, 5.0)
        self.assertLess(orbit.pitch, 90.0)
        top: float = orbit.pitch
        orbit.turn(0.0, 5.0)
        self.assertEqual(orbit.pitch, top)                              # held at its limit
        orbit.dolly(-10.0)
        closest: float = orbit.zoom
        orbit.dolly(-10.0)
        self.assertEqual(orbit.zoom, closest)
        self.assertGreater(closest, 0.0)

    def test_it_knows_the_side_it_is_on(self) -> None:
        orbit = Orbit()
        self.assertTrue(orbit.front_on_near_side())
        orbit.turn(-1.0, 0.0)                                           # 180° round
        self.assertFalse(orbit.front_on_near_side())
        self.assertTrue(math.isclose(abs(orbit.yaw), 180.0))


if __name__ == '__main__':
    unittest.main()
