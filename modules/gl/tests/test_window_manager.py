"""Tests for the window loop: its end from another thread, its survival of what its event callbacks raise, and a
monitor's hot-plug at the top of the next frame (glfw mocked: no window, no GL)."""

import ctypes
import sys
import threading
import time
import unittest
from unittest import mock

from modules.gl import Monitor, WindowManager, WindowManagerSettings, WindowSettings

MODULE = sys.modules['modules.gl.window_manager']     # the module: the package exports the class under its name
LOGGER: str = 'modules.gl.window_manager'
DISCONNECTED: int = 0x00040002


class TestWindowManager(unittest.TestCase):
    def setUp(self):
        self.renderer = mock.Mock()
        self.manager = WindowManager(self.renderer, WindowManagerSettings(), [WindowSettings(), WindowSettings()])
        patcher = mock.patch.object(MODULE, 'glfw')
        self.glfw = patcher.start()
        self.addCleanup(patcher.stop)
        self.glfw.DISCONNECTED = DISCONNECTED
        self.glfw.window_should_close.return_value = False

    def with_windows(self) -> list[mock.Mock]:
        windows = [mock.Mock(index=i, shown=True) for i in range(2)]
        for window in windows:
            window.framebuffer_size.return_value = (100, 100)
        self.manager._windows = windows
        self.manager._sizes = [(100, 100), (100, 100)]
        return windows

    def run_loop(self, frames: int) -> None:
        """The render loop for that many frames, without a window: update and draw do nothing."""
        if not self.manager._windows:
            self.with_windows()
        count: list[int] = [0]

        def should_close(_handle) -> bool:
            count[0] += 1
            return count[0] > frames * len(self.manager._windows)
        self.glfw.window_should_close.side_effect = should_close
        with mock.patch.object(self.manager, '_update'), mock.patch.object(self.manager, '_draw'):
            self.manager._render_loop()

    def test_stop_from_another_thread_ends_the_loop(self):
        frames: list[int] = []
        limit: int = 5000                             # the loop's end without stop(): a failed test, not a hung one
        self.glfw.window_should_close.side_effect = lambda _: len(frames) >= limit
        self.glfw.poll_events.side_effect = lambda: (frames.append(1), time.sleep(0.001))
        stopper = threading.Timer(0.05, self.manager.stop)

        def setup() -> None:
            self.with_windows()
            stopper.start()

        with mock.patch.object(self.manager, '_setup', side_effect=setup), mock.patch.object(self.manager, '_update'), \
                mock.patch.object(self.manager, '_draw'):
            self.manager.run()
        stopper.join(timeout=3.0)
        self.assertFalse(stopper.is_alive())          # stop() returned: it saw the loop end
        self.assertLess(len(frames), limit)
        self.assertIs(self.manager._loop_thread, threading.current_thread())
        self.glfw.post_empty_event.assert_called()
        self.renderer.deallocate.assert_called_once()

    def test_an_exception_from_an_event_callback_does_not_end_the_loop(self):
        # pyGLFW raises what a callback raised from poll_events
        self.glfw.poll_events.side_effect = ValueError('NULL pointer access')
        with self.assertLogs(LOGGER, 'ERROR'):
            self.run_loop(3)
        self.assertEqual(self.glfw.poll_events.call_count, 3)
        self.renderer.deallocate.assert_called_once()

    def test_every_frame_starts_in_window_0s_context(self):
        # drawing a window leaves its context current; the renderer's objects live in window 0's
        windows = self.with_windows()
        current: list[object] = []
        for window in windows:
            window.make_current.side_effect = lambda window=window: current.append(window)
        self.renderer.update.side_effect = lambda: current.append(('update', current[-1]))
        with mock.patch.object(self.manager, '_tick_windows'), mock.patch.object(MODULE, 'glBindFramebuffer'), \
                mock.patch.object(MODULE, 'glViewport'):
            count: list[int] = [0]
            self.glfw.window_should_close.side_effect = lambda _h: (count.__setitem__(0, count[0] + 1), count[0] > 4)[1]
            self.manager._render_loop()
        self.assertEqual([c for c in current if isinstance(c, tuple)], [('update', windows[0])] * 2)
        self.assertIn(windows[1], current)                                    # window 1 drawn and swapped

    def test_the_v_sync_window_is_swapped_last(self):
        windows = self.with_windows()
        self.manager.settings.v_sync_window = 0
        swapped: list[object] = []
        self.glfw.swap_buffers.side_effect = swapped.append
        with mock.patch.object(MODULE, 'glBindFramebuffer'), mock.patch.object(MODULE, 'glViewport'):
            self.manager._draw()
        self.assertEqual(swapped, [windows[1].handle, windows[0].handle])

    def test_a_hot_plug_is_handled_at_the_top_of_the_next_frame(self):
        self.with_windows()
        with mock.patch.object(MODULE, 'Monitor') as monitor_class:
            monitor_class.connected.return_value = [Monitor('m', 0, 0, 1920, 1080, ctypes.pointer(ctypes.c_int(1)))]
            self.manager._on_monitor_change(object(), DISCONNECTED)
            monitor_class.connected.assert_not_called()          # the callback only marks it
            self.run_loop(1)
            self.run_loop(1)
        monitor_class.connected.assert_called_once()
        self.assertEqual(self.manager.settings.connected, 'M0 m 1920×1080')

    def pace(self, times: list[float]) -> list[float]:
        """The deadlines the pacing waits for, one frame ending at each of `times`."""
        waits: list[float] = []
        clock = iter(times)
        with mock.patch.object(MODULE, 'perf_counter', lambda: next(clock)), \
                mock.patch.object(self.manager, '_waker') as waker:
            waker.wait.side_effect = lambda deadline: waits.append(round(deadline, 6))
            for _ in times:
                self.manager._pace()
        return waits

    def test_v_sync_on_a_shown_window_paces_the_loop_itself(self):
        self.with_windows()
        self.manager.settings.fps_limit = 100.0
        self.assertEqual(self.pace([0.0, 0.01]), [])

    def test_fps_limit_waits_for_an_absolute_deadline_and_resyncs_when_a_period_behind(self):
        self.with_windows()
        self.manager.settings.v_sync = False
        self.manager.settings.fps_limit = 100.0
        # on time, on time, past its deadline (on at once, the deadline kept), more than a period behind (moved to
        # now, no wait), then on time from there
        self.assertEqual(self.pace([0.0, 0.0101, 0.0335, 0.06, 0.065]), [0.01, 0.02, 0.07])
        self.assertAlmostEqual(self.manager._next_frame, 0.07)

    def test_every_window_hidden_runs_at_60_fps_without_a_limit(self):
        windows = self.with_windows()
        for window in windows:
            window.shown = False
        self.assertEqual(self.pace([0.0, 1 / 60]), [round(1 / 60, 6), round(2 / 60, 6)])
        windows[0].shown = True
        self.manager.settings.v_sync = False
        self.assertEqual(self.pace([1.0]), [])                                # shown, no v-sync, no limit: free

    def test_f_toggles_the_fullscreen_of_the_window_with_focus_and_esc_closes(self):
        F, ESCAPE, PRESS = 70, 256, 1
        self.glfw.KEY_F, self.glfw.KEY_ESCAPE, self.glfw.PRESS = F, ESCAPE, PRESS
        self.manager._on_key(1, F, PRESS, 0)
        self.assertEqual([s.fullscreen for s in self.manager.window_settings], [False, True])
        self.assertFalse(self.manager._close_request.is_set())
        self.manager._on_key(0, ESCAPE, PRESS, 0)
        self.assertTrue(self.manager._close_request.is_set())


if __name__ == '__main__':
    unittest.main()
