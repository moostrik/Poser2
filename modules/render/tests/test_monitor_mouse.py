"""The monitor's mouse: a scroll steps one of the show's layers and stays inside the volume; a drag
turns its orbit; a reset brings the front view at the middle layer back. Both write the monitor's
settings, so the panel shows them and a preset keeps them."""

import unittest

from modules.gl import MouseButton
from modules.render.layers import MonitorLayer, MonitorLayerSettings


class MonitorMouseTest(unittest.TestCase):
    def test_a_scroll_steps_one_layer_and_stays_inside(self) -> None:
        monitor = MonitorLayer(MonitorLayerSettings())
        advanced = monitor.settings.advanced
        monitor.scroll(1, 16)
        self.assertAlmostEqual(advanced.slice_position, 9.5 / 16)
        monitor.scroll(100, 16)
        self.assertAlmostEqual(advanced.slice_position, 15.5 / 16)
        monitor.scroll(-100, 16)
        self.assertAlmostEqual(advanced.slice_position, 0.5 / 16)

    def test_a_drag_turns_and_a_reset_brings_the_front_view_back(self) -> None:
        monitor = MonitorLayer(MonitorLayerSettings())
        orbit = monitor.settings.advanced.orbit
        monitor.drag(0.2, 0.1, MouseButton.LEFT)
        monitor.drag(0.0, -0.2, MouseButton.RIGHT)
        self.assertNotEqual((orbit.yaw, orbit.zoom), (0.0, 1.0))
        monitor.scroll(3, 16)
        monitor.reset_view()
        self.assertEqual((orbit.yaw, orbit.pitch, orbit.zoom, monitor.settings.advanced.slice_position), (0.0, 0.0, 1.0, 0.5))


if __name__ == '__main__':
    unittest.main()
