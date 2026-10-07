"""Tests for the window manager's survival of what its event callbacks raise: a monitor that is
unplugged must not end the app (glfw mocked: no window, no GL)."""

import ctypes
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from modules.gl import FullscreenMode, WindowManager, WindowSettings

MODULE = sys.modules['modules.gl.WindowManager']      # the module: the package exports the class under its name
LOGGER: str = 'modules.gl.WindowManager'
DISCONNECTED: int = 0x00040002


class TestWindowManager(unittest.TestCase):
    def setUp(self):
        self.renderer = mock.Mock()
        self.manager = WindowManager(self.renderer, WindowSettings())
        patcher = mock.patch.object(MODULE, 'glfw')
        self.glfw = patcher.start()
        self.addCleanup(patcher.stop)
        self.glfw.DISCONNECTED = DISCONNECTED

    def run_loop(self, frames: int) -> None:
        """The render loop for that many frames, without a window: update and draw do nothing."""
        self.manager._main_window = object()
        self.glfw.window_should_close.side_effect = [False] * frames + [True]
        with mock.patch.object(self.manager, '_bind_quad_vao'), mock.patch.object(self.manager, '_update'), \
                mock.patch.object(self.manager, '_draw_main_window'):
            self.manager._render_loop()

    def test_an_exception_from_an_event_callback_does_not_end_the_loop(self):
        # pyGLFW raises what a callback raised from poll_events
        self.glfw.poll_events.side_effect = ValueError('NULL pointer access')
        with self.assertLogs(LOGGER, 'ERROR'):
            self.run_loop(3)
        self.assertEqual(self.glfw.poll_events.call_count, 3)
        self.renderer.deallocate.assert_called_once()

    def test_an_exception_from_a_deferred_call_does_not_end_the_loop(self):
        done: list[int] = []
        self.manager._deferred.put(lambda: 1 / 0)
        self.manager._deferred.put(lambda: done.append(1))
        with self.assertLogs(LOGGER, 'ERROR'):
            self.run_loop(2)
        self.assertEqual(done, [1])                 # the calls after it still run
        self.assertEqual(self.glfw.poll_events.call_count, 2)

    def test_a_monitor_without_a_video_mode_is_skipped(self):
        # a monitor that is being unplugged is still listed, and asking for its video mode fails
        gone, good = object(), object()
        self.glfw.get_window_size.return_value = (100, 100)
        self.glfw.get_monitors.return_value = [gone, good]
        self.glfw.get_monitor_pos.return_value = (0, 0)

        def video_mode(monitor):
            if monitor is gone:
                raise ValueError('NULL pointer access')
            return SimpleNamespace(size=SimpleNamespace(width=1920, height=1080))

        self.glfw.get_video_mode.side_effect = video_mode
        self.assertEqual(self.manager.fullscreen_mode, FullscreenMode.WINDOWED)
        self.manager._window_pos_callback(object(), 10, 20)
        self.assertIs(self.manager._monitor, good)
        self.assertEqual((self.manager.settings.x, self.manager.settings.y), (10, 20))

    def test_the_handle_of_a_disconnected_monitor_is_dropped(self):
        self.glfw.get_monitors.return_value = []
        ours = ctypes.pointer(ctypes.c_int(1))
        other = ctypes.pointer(ctypes.c_int(2))
        self.manager._monitor = ours
        self.manager._on_monitor_change(other, DISCONNECTED)
        self.assertIs(self.manager._monitor, ours)
        # GLFW hands the callback a handle object of its own for the same monitor
        self.manager._on_monitor_change(ctypes.pointer(ours.contents), DISCONNECTED)
        self.assertIsNone(self.manager._monitor)


if __name__ == '__main__':
    unittest.main()
