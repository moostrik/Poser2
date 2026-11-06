"""The monitor's mouse: a scroll steps one of the show's layers and stays inside the volume; a drag
turns its orbit; a reset brings the front view at the middle layer back."""

import unittest

from modules.gl import MouseButton
from modules.render.layers import MonitorLayer, MonitorLayerSettings


class MonitorMouseTest(unittest.TestCase):
    def test_a_scroll_steps_one_layer_and_stays_inside(self) -> None:
        monitor = MonitorLayer(MonitorLayerSettings())
        monitor.scroll(1, 16)
        self.assertAlmostEqual(monitor._slice, 9.5 / 16)
        monitor.scroll(100, 16)
        self.assertAlmostEqual(monitor._slice, 15.5 / 16)
        monitor.scroll(-100, 16)
        self.assertAlmostEqual(monitor._slice, 0.5 / 16)

    def test_a_drag_turns_and_a_reset_brings_the_front_view_back(self) -> None:
        monitor = MonitorLayer(MonitorLayerSettings())
        monitor.drag(0.2, 0.1, MouseButton.LEFT)
        monitor.drag(0.0, -0.2, MouseButton.RIGHT)
        self.assertNotEqual((monitor._orbit.yaw, monitor._orbit.zoom), (0.0, 1.0))
        monitor.scroll(3, 16)
        monitor.reset_view()
        self.assertEqual((monitor._orbit.yaw, monitor._orbit.pitch, monitor._orbit.zoom, monitor._slice), (0.0, 0.0, 1.0, 0.5))


if __name__ == '__main__':
    unittest.main()
