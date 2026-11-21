"""Tests for the windows' input callbacks: a raising one does not stop the others (glfw mocked)."""

import sys
import unittest
from unittest import mock

from modules.window.window_input import WindowInput

MODULE = sys.modules['modules.window.window_input']
LOGGER: str = 'modules.utils.broadcast'      # a raising callback is logged by its Broadcast


class TestWindowInput(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(MODULE, 'glfw')
        self.glfw = patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_raising_callback_is_logged_and_the_next_one_runs(self):
        window_input = WindowInput()
        heard: list[tuple[int, int]] = []

        def raising(window: int, key: int, action: int, mods: int) -> None:
            raise ValueError('a bug in a callback')

        window_input.add_key_callback(raising)
        window_input.add_key_callback(lambda window, key, action, mods: heard.append((window, key)))
        with self.assertLogs(LOGGER, 'ERROR'):
            window_input._on_key(2, 70, 1, 0)
        self.assertEqual(heard, [(2, 70)])

    def test_the_mouse_is_told_its_window_and_its_place_over_it(self):
        window_input = WindowInput()
        self.glfw.get_window_size.return_value = (200, 100)
        moves: list[tuple[int, float, float]] = []
        window_input.add_mouse_move_callback(lambda window, x, y: moves.append((window, x, y)))
        window_input._on_cursor(1, object(), 50.0, 25.0)
        self.assertEqual(moves, [(1, 0.25, 0.25)])


if __name__ == '__main__':
    unittest.main()
