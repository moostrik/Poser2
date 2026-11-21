"""Tests for a window's settings applied at the tick, and a user's move written back once (glfw mocked)."""

import ctypes
import sys
import unittest
from unittest import mock

from modules.window import WindowSettings
from modules.window.monitor import Monitor, MonitorId
from modules.window.window import Window

MODULE = sys.modules['modules.window.window']


def monitor(name: str, x: int, y: int, width: int = 1920, height: int = 1080) -> Monitor:
    return Monitor(name, x, y, width, height, ctypes.pointer(ctypes.c_int(x)))


class TestWindow(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(MODULE, 'glfw')
        self.glfw = patcher.start()
        self.addCleanup(patcher.stop)
        self.glfw.get_window_attrib.return_value = 0          # not minimized
        self.monitors = [monitor('left', 0, 0), monitor('right', 1920, 0)]
        self.settings = WindowSettings()
        self.window = Window(0, self.settings, 'test', None, lambda: None)

    def tick(self) -> None:
        """What the window manager does with a window at the tick."""
        self.window.write_move(self.monitors, False)
        self.window.apply(self.window.placement(self.monitors, False))

    def test_a_setting_change_is_applied_once(self):
        self.tick()
        self.assertEqual(self.glfw.set_window_monitor.call_count, 1)        # the first placement
        self.settings.x = 300
        self.tick()
        self.tick()
        self.assertEqual(self.glfw.set_window_monitor.call_count, 2)
        self.glfw.set_window_monitor.assert_called_with(self.window.handle, None, 300, 40, 1280, 720, 0)
        self.settings.enabled = False                                        # hidden: shown again only on a change
        self.tick()
        self.tick()
        self.glfw.hide_window.assert_called_once()
        self.assertEqual(self.glfw.set_window_monitor.call_count, 2)

    def test_a_drag_is_written_once_and_not_applied_back(self):
        self.tick()
        self.glfw.set_window_monitor.reset_mock()
        # the user drags the window onto the right monitor and resizes it: GLFW's callbacks, then the tick
        self.glfw.get_window_pos.return_value = (2000, 50)
        self.glfw.get_window_size.return_value = (800, 600)
        self.window._on_move(self.window.handle, 2000, 50)
        self.window._on_move(self.window.handle, 800, 600)
        self.tick()
        s = self.settings
        self.assertEqual((s.monitor, s.x, s.y, s.width, s.height), (MonitorId.M1, 80, 50, 800, 600))
        self.glfw.set_window_monitor.assert_not_called()
        self.tick()                                                           # nothing more to write or apply
        self.glfw.set_window_monitor.assert_not_called()

    def test_our_own_placement_is_not_taken_for_the_users_move(self):
        self.glfw.set_window_monitor.side_effect = lambda *args: self.window._on_move(self.window.handle, 0, 0)
        self.tick()
        self.assertFalse(self.window._moved)

    def test_a_window_whose_monitor_is_missing_waits_small_and_keeps_its_settings(self):
        self.settings.monitor = MonitorId.M3
        self.settings.fullscreen = True
        placement = self.window.placement(self.monitors, False)
        self.assertEqual((placement.x, placement.y, placement.width, placement.height), (40, 40, 200, 200))
        self.assertTrue(placement.decorated)
        self.assertEqual((self.settings.monitor, self.settings.fullscreen), (MonitorId.M3, True))

    def test_the_mouse_is_hidden_in_fullscreen_unless_the_setting_shows_it(self):
        self.settings.fullscreen = True
        self.assertTrue(self.window.placement(self.monitors, False).cursor_hidden)
        self.assertFalse(self.window.placement(self.monitors, True).cursor_hidden)
        self.settings.fullscreen = False                                      # a window always shows it
        self.assertFalse(self.window.placement(self.monitors, False).cursor_hidden)


if __name__ == '__main__':
    unittest.main()
