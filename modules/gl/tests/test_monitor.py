"""Tests for the connected monitors' order and the monitors left out (glfw mocked)."""

import ctypes
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from modules.gl.monitor import Monitor

MODULE = sys.modules['modules.gl.monitor']


def handle(n: int):
    return ctypes.pointer(ctypes.c_int(n))


class TestMonitor(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(MODULE, 'glfw')
        self.glfw = patcher.start()
        self.addCleanup(patcher.stop)
        self.positions: dict[int, tuple[int, int]] = {}
        self.glfw.get_monitor_pos.side_effect = lambda h: self.positions[ctypes.addressof(h.contents)]
        self.glfw.get_monitor_name.side_effect = lambda h: f'm{h.contents.value}'.encode()
        self.glfw.get_video_mode.return_value = SimpleNamespace(size=SimpleNamespace(width=1920, height=1080))

    def place(self, h, x: int, y: int) -> None:
        self.positions[ctypes.addressof(h.contents)] = (x, y)

    def test_the_primary_first_then_by_x_then_y(self):
        right, primary, left_low, left_high = handle(1), handle(2), handle(3), handle(4)
        self.place(right, 3840, 0)
        self.place(primary, 1920, 0)
        self.place(left_low, 0, 500)
        self.place(left_high, 0, -1080)
        self.glfw.get_monitors.return_value = [right, primary, left_low, left_high]
        self.glfw.get_primary_monitor.return_value = ctypes.pointer(primary.contents)    # a handle object of its own
        names = [m.name for m in Monitor.connected()]
        self.assertEqual(names, ['m2', 'm4', 'm3', 'm1'])

    def test_a_monitor_without_a_video_mode_is_skipped(self):
        # a monitor that is being unplugged is still listed, and asking for its video mode fails or gives none
        raising, empty, good = handle(1), handle(2), handle(3)
        for h in (raising, empty, good):
            self.place(h, 0, 0)
        self.glfw.get_monitors.return_value = [raising, empty, good]
        self.glfw.get_primary_monitor.return_value = good

        def video_mode(h):
            if h is raising:
                raise ValueError('NULL pointer access')
            if h is empty:
                return None
            return SimpleNamespace(size=SimpleNamespace(width=2560, height=1440))

        self.glfw.get_video_mode.side_effect = video_mode
        monitors = Monitor.connected()
        self.assertEqual([m.name for m in monitors], ['m3'])
        self.assertEqual((monitors[0].width, monitors[0].height), (2560, 1440))


if __name__ == '__main__':
    unittest.main()
