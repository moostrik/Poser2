"""Tests for the online cell of the walls: what it shows, and that the player's bang and caption skip it (no GL, no OSC)."""

import unittest
from unittest import mock

from apps.smile.player.player import Player, online_cell_mode


class TestOnlineCellMode(unittest.TestCase):
    def test_a_clip_while_off_or_without_the_relay_the_card_while_free_the_screen_during_a_turn(self):
        self.assertEqual(online_cell_mode(False, True, True), 'clip')
        self.assertEqual(online_cell_mode(True, False, False), 'clip')
        self.assertEqual(online_cell_mode(True, True, False), 'qr')
        self.assertEqual(online_cell_mode(True, True, True), 'screen')


class TestPlaysOnTheOnlineCell(unittest.TestCase):
    def setUp(self):
        # a player without its threads: the loader callbacks are called directly
        self.walls = [mock.Mock(), mock.Mock()]
        self.clip_player = Player.__new__(Player)
        self.clip_player.walls = self.walls
        self.clip_player.osc = [mock.Mock(), mock.Mock()]
        self.clip_player._lines = mock.Mock(pick=lambda: 'a line')
        self.clip_player._lines_lock = mock.MagicMock()
        self.clip_player.online_cell = None

    def test_a_play_bangs_max_and_captions_the_cell_on_its_wall(self):
        self.clip_player._on_play(30)                    # wall 1, cell 6
        self.clip_player.osc[1].submit.assert_called_once_with('/bang', 6)
        self.walls[1].set_caption.assert_called_once_with('6', 'a line')
        self.clip_player.osc[0].submit.assert_not_called()

    def test_the_online_cells_play_is_nobodys(self):
        self.clip_player.online_cell = 30
        self.clip_player._on_play(30)
        self.clip_player.osc[1].submit.assert_not_called()
        self.walls[1].set_caption.assert_not_called()
        self.clip_player._on_play(7)                     # another cell is as before
        self.clip_player.osc[0].submit.assert_called_once_with('/bang', 7)


if __name__ == '__main__':
    unittest.main()
