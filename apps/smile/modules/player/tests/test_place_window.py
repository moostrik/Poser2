"""Tests for where a player opens its window."""

import unittest

from modules.player.Player import place_window


class TestPlaceWindow(unittest.TestCase):
    def test_own_monitor_keeps_the_mode(self):
        self.assertEqual(place_window(1, False, 3), (1, False))
        self.assertEqual(place_window(2, False, 3), (2, False))
        self.assertEqual(place_window(0, True, 1), (0, True))

    def test_missing_monitor_is_a_window_on_the_primary(self):
        self.assertEqual(place_window(2, False, 2), (0, True))
        self.assertEqual(place_window(1, False, 1), (0, True))
        self.assertEqual(place_window(2, True, 1), (0, True))


if __name__ == '__main__':
    unittest.main()
