"""Tests for the scrolling name: the sphere's rows, a pass's path over both players, the name's position on
every screen, the strip image, and the scroller that plans and sends the passes (no GL, no players)."""

import random
import unittest
from types import SimpleNamespace

import numpy as np

from modules.credit_scroll import (ADDRESS, LOOP_ADDRESS, SCREEN_PX, CreditScroller, ScrollPass, name_strip, parse_rows,
                                   plan_loops, plan_pass, row_screens, screen_cell, scroll_done, scroll_offset)


class TestRows(unittest.TestCase):
    def test_the_middle_rows_have_twelve_screens_and_the_outer_ones_six(self):
        self.assertEqual(row_screens(1), [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45])
        self.assertEqual(row_screens(3), [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47])
        self.assertEqual(row_screens(0), [0, 8, 16, 24, 32, 40])
        self.assertEqual(row_screens(4), [4, 12, 20, 28, 36, 44])
        # every screen is in exactly one row
        self.assertEqual(sorted(s for row in range(5) for s in row_screens(row)), list(range(48)))

    def test_a_screen_is_a_cell_of_a_player_by_the_wiring(self):
        self.assertEqual(screen_cell(0), (0, 0))
        self.assertEqual(screen_cell(1), (0, 6))     # screen 2: unit 1 output 2, grid row 1
        self.assertEqual(screen_cell(4), (0, 1))     # screen 5: unit 2 output 1
        self.assertEqual(screen_cell(23), (0, 23))
        self.assertEqual(screen_cell(24), (1, 0))
        self.assertEqual(screen_cell(47), (1, 23))

    def test_the_rows_setting_is_parsed_leniently(self):
        self.assertEqual(parse_rows("1,3"), [1, 3])
        self.assertEqual(parse_rows(" 3 1 1"), [3, 1])
        self.assertEqual(parse_rows("5, x, -1, 2"), [2])
        self.assertEqual(parse_rows(""), [])


class TestPass(unittest.TestCase):
    def test_the_path_follows_the_ring_and_wraps(self):
        # down the ribs: the ribs count to the camera's left, the visitor's right, and the name moves left
        self.assertEqual(ScrollPass(1, 2, 3).screens, [9, 5, 1])
        self.assertEqual(ScrollPass(1, 1, 4).screens, [5, 1, 45, 41])
        self.assertEqual(ScrollPass(1, 10, 3, reverse=True).screens, [41, 45, 1])

    def test_each_player_gets_its_own_cells_with_their_place_on_the_whole_path(self):
        scroll = ScrollPass(1, 4, 6)      # screens (from 0) 17, 13, 9, 5, 1 | 45: height 1, grid row 1
        self.assertEqual(scroll.cells(0), [(10, 0), (9, 1), (8, 2), (7, 3), (6, 4)])
        self.assertEqual(scroll.cells(1), [(11, 5)])

    def test_the_readout_numbers_the_screens_from_one(self):
        self.assertEqual(ScrollPass(3, 0, 2).describe(), "row 3: screens 4, 48")

    def test_a_pass_is_planned_within_the_settings(self):
        rng = random.Random(1)
        for _ in range(50):
            scroll = plan_pass([1, 3], 6, 10, False, rng)
            assert scroll is not None
            self.assertIn(scroll.row, (1, 3))
            self.assertTrue(0 <= scroll.start < 12)
            self.assertTrue(6 <= scroll.count <= 10)
        self.assertIsNone(plan_pass([], 6, 10, False, rng))
        # a count over the ring's screens is the ring; min and max the wrong way round still work
        self.assertEqual(plan_pass([0], 10, 12, False, rng).count, 6)
        self.assertTrue(6 <= plan_pass([1], 10, 6, False, rng).count <= 10)


class TestLoops(unittest.TestCase):
    def test_each_row_gets_a_word_and_the_whole_ring(self):
        loops = plan_loops([1, 3], ["Matthias", "Oostrik"], False)
        self.assertEqual([(word, scroll.row, scroll.count) for word, scroll in loops], [("Matthias", 1, 12), ("Oostrik", 3, 12)])
        self.assertEqual(sorted(loops[0][1].screens), row_screens(1))
        self.assertEqual([word for word, _ in plan_loops([0, 1, 2], ["One"], False)], ["One"] * 3)
        self.assertEqual(plan_loops([1], [], False), [])
        self.assertEqual(plan_loops([], ["A"], False), [])


class TestPosition(unittest.TestCase):
    def test_the_name_enters_the_first_screen_at_its_right_edge_and_moves_left(self):
        # before moving, the strip's x = 0 is at the screen's right edge: the screen shows nothing of it yet
        self.assertEqual(scroll_offset(0, 0.0, 0.0), -SCREEN_PX)
        # after a screen's width, the strip's start is at the screen's left edge
        self.assertEqual(scroll_offset(0, SCREEN_PX, 0.0), 0.0)
        # the next screen is one screen and a gap further along
        self.assertEqual(scroll_offset(1, SCREEN_PX, 100.0), -SCREEN_PX - 100.0)

    def test_a_pass_is_done_when_the_end_of_the_strip_has_left_the_last_screen(self):
        strip = 3000
        count, gap = 8, 0.0
        last_left_edge = (count - 1) * SCREEN_PX + SCREEN_PX
        self.assertFalse(scroll_done(last_left_edge + strip - 1, count, strip, gap))
        self.assertTrue(scroll_done(last_left_edge + strip, count, strip, gap))


class TestStrip(unittest.TestCase):
    def test_the_strip_is_the_screen_high_with_the_ink_the_size_in_the_middle(self):
        strip = name_strip("Matthias Oostrik", 640, 512)
        self.assertEqual(strip.shape[0], 640)
        self.assertEqual(strip.shape[2], 4)
        alpha = strip[:, :, 3]
        rows = np.flatnonzero(alpha.max(axis=1))
        self.assertTrue(rows.size)
        self.assertGreaterEqual(rows[0], 64 - 1)
        self.assertLessEqual(rows[-1], 640 - 64)
        # nearly as tall as the screen: the letters with their edge fill the inner height
        self.assertGreater(rows[-1] - rows[0], 0.9 * (640 - 2 * 64))
        # in the vertical middle
        self.assertLessEqual(abs(rows[0] - (639 - rows[-1])), 3)
        # white letters on a black edge: both colours are there where the alpha is
        self.assertTrue((strip[:, :, :3][alpha > 0] == 255).any())
        self.assertTrue((strip[:, :, :3][alpha > 0] == 0).any())
        # several screens long
        self.assertGreater(strip.shape[1], 6 * SCREEN_PX)

    def test_a_bigger_size_makes_a_longer_name_and_the_screen_is_the_most(self):
        self.assertGreater(name_strip("Matthias Oostrik", 640, 640).shape[1], name_strip("Matthias Oostrik", 640, 440).shape[1])
        self.assertEqual(name_strip("Matthias", 640, 900).shape[1], name_strip("Matthias", 640, 640).shape[1])

    def test_an_empty_name_is_an_empty_strip(self):
        self.assertEqual(name_strip("", 640, 512).shape, (640, 1, 4))


def settings(**values) -> SimpleNamespace:
    base = dict(credit_name="Matthias Oostrik", credit_scroll=True, credit_scroll_every=120, credit_scroll_rows="1,3",
                credit_scroll_screens_min=6, credit_scroll_screens_max=10, credit_scroll_speed=1.0, credit_scroll_size=512,
                credit_scroll_gap=0, credit_scroll_reverse=False, credit_scroll_loop=False, credit_scroll_last='x')
    base.update(values)
    return SimpleNamespace(**base)


class TestScroller(unittest.TestCase):
    def setUp(self):
        self.sent: list[tuple[int, str, list]] = []
        self.settings = settings()
        self.scroller = CreditScroller(self.settings, lambda player, address, args: self.sent.append((player, address, args)))

    def test_the_readout_starts_empty(self):
        self.assertEqual(self.settings.credit_scroll_last, '')

    def test_a_pass_goes_out_every_interval_while_on(self):
        self.scroller.update(1000.0, False)
        self.assertEqual(self.sent, [])
        self.scroller.update(1119.0, False)
        self.assertEqual(self.sent, [])
        self.scroller.update(1120.0, False)
        self.assertTrue(self.sent)
        self.assertIn('row', self.settings.credit_scroll_last)
        self.sent.clear()
        self.scroller.update(1200.0, False)
        self.assertEqual(self.sent, [])
        self.scroller.update(1240.0, False)
        self.assertTrue(self.sent)

    def test_the_message_carries_one_word_of_the_name_the_speed_in_px_and_each_players_cells(self):
        random.seed(3)
        self.scroller.send_pass(0.0)
        players = {player for player, _, _ in self.sent}
        self.assertTrue(players)
        for player, address, args in self.sent:
            self.assertEqual(address, ADDRESS)
            self.assertIn(args[0], ("Matthias", "Oostrik"))
            self.assertEqual(args[1:4], [512.0, 0, 512])
            pairs = args[4:]
            self.assertEqual(len(pairs) % 2, 0)
            self.assertTrue(all(0 <= cell < 24 for cell in pairs[::2]))
            self.assertTrue(all(0 <= place < 10 for place in pairs[1::2]))
        places = sorted(place for _, _, args in self.sent for place in args[5::2])
        self.assertEqual(places, list(range(len(places))))

    def test_the_passes_take_the_words_of_the_name_by_turns(self):
        words = []
        for _ in range(4):
            self.sent.clear()
            self.scroller.send_pass(0.0)
            words.append({args[0] for _, _, args in self.sent}.pop())
        self.assertEqual(words[0], words[2])
        self.assertEqual(words[1], words[3])
        self.assertEqual(set(words), {"Matthias", "Oostrik"})
        self.assertIn(words[-1], self.settings.credit_scroll_last)

    def test_a_one_word_name_is_sent_whole(self):
        self.settings.credit_name = "  Banksy "
        self.scroller.send_pass(0.0)
        self.assertTrue(all(args[0] == "Banksy" for _, _, args in self.sent))

    def test_the_button_sends_one_pass_at_once_and_the_switch_off_sends_none(self):
        self.settings.credit_scroll = False
        self.scroller.update(10.0, False)
        self.scroller.update(10000.0, False)
        self.assertEqual(self.sent, [])
        self.scroller.request()
        self.scroller.update(10001.0, False)
        self.assertEqual(len(self.sent), len({p for p, _, _ in self.sent}))

    def test_nothing_goes_out_while_blocked_and_the_timer_moves_on(self):
        self.scroller.update(0.0, False)
        self.scroller.request()
        self.scroller.update(120.0, True)
        self.assertEqual(self.sent, [])
        self.scroller.update(121.0, False)
        self.assertEqual(self.sent, [])
        self.scroller.update(240.0, False)
        self.assertTrue(self.sent)

    def test_the_continuous_mode_sends_the_loops_every_second_with_the_distance_travelled(self):
        self.settings.credit_scroll_loop = True
        self.scroller.update(100.0, False)
        self.scroller.update(100.5, False)
        loops = [(player, args) for player, address, args in self.sent if address == LOOP_ADDRESS]
        self.assertEqual(len(loops), 4)        # two rows, both players
        self.assertEqual({args[0] for _, args in loops}, {"Matthias", "Oostrik"})
        for _, args in loops:
            self.assertEqual(args[1:6], [512.0, 0, 512, 12, 0.0])
            self.assertEqual(len(args[6::2]), 6)                   # this player's half of the ring
            self.assertTrue(set(args[7::2]) < set(range(12)))      # with their places round it
        self.assertEqual(self.settings.credit_scroll_last, 'continuous: Matthias row 1, Oostrik row 3')
        self.sent.clear()
        self.scroller.update(101.0, False)
        self.assertEqual({args[5] for _, _, args in self.sent}, {512.0})
        # no timed passes meanwhile, but the button still works
        self.sent.clear()
        self.scroller.update(300.0, False)
        self.assertFalse([1 for _, address, _ in self.sent if address == ADDRESS])
        self.scroller.request()
        self.scroller.update(300.1, False)
        self.assertTrue([1 for _, address, _ in self.sent if address == ADDRESS])

    def test_the_loops_stop_while_blocked_and_start_again_from_zero(self):
        self.settings.credit_scroll_loop = True
        self.scroller.update(0.0, False)
        self.scroller.update(10.0, True)
        self.assertFalse([1 for _, _, args in self.sent if args[5] == 10.0 * 512])
        self.sent.clear()
        self.scroller.update(20.0, False)
        self.assertEqual({args[5] for _, _, args in self.sent}, {0.0})

    def test_no_valid_row_is_a_warning_not_a_pass(self):
        self.settings.credit_scroll_rows = "7"
        with self.assertLogs('modules.credit_scroll', 'WARNING'):
            self.assertIsNone(self.scroller.send_pass(0.0))
        self.assertEqual(self.sent, [])

    def test_a_player_that_cannot_be_reached_does_not_stop_the_other(self):
        def send(player, address, args):
            if player == 0:
                raise OSError('no player')
            self.sent.append((player, address, args))
        scroller = CreditScroller(self.settings, send)
        random.seed(0)
        for _ in range(20):
            with self.assertLogs('modules.credit_scroll'):
                scroller.send_pass(0.0)
        self.assertTrue(all(player == 1 for player, _, _ in self.sent))


if __name__ == '__main__':
    unittest.main()
