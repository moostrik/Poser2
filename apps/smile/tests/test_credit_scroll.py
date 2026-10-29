"""Tests for the scrolling name: the sphere's rows, a word's path round a ring over both walls, the word's
position on every screen, the strip image, and the scroller that hands the loops to the walls (no GL)."""

import unittest
from types import SimpleNamespace

import numpy as np

from apps.smile.credit_scroll import (SCREEN_PX, CreditScroller, RingPath, name_strip, parse_rows, plan_loops,
                                   row_screens, screen_cell, scroll_offset)


class TestRows(unittest.TestCase):
    def test_the_middle_rows_have_twelve_screens_and_the_outer_ones_six(self):
        self.assertEqual(row_screens(1), [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45])
        self.assertEqual(row_screens(3), [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47])
        self.assertEqual(row_screens(0), [0, 8, 16, 24, 32, 40])
        self.assertEqual(row_screens(4), [4, 12, 20, 28, 36, 44])
        # every screen is in exactly one row
        self.assertEqual(sorted(s for row in range(5) for s in row_screens(row)), list(range(48)))

    def test_a_screen_is_a_cell_of_a_wall_by_the_wiring(self):
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


class TestPath(unittest.TestCase):
    def test_the_path_goes_down_the_ribs_from_the_camera_and_round(self):
        # the ribs count to the camera's left, the visitor's right, and the word moves to the visitor's left
        self.assertEqual(RingPath(1).screens, [1, 45, 41, 37, 33, 29, 25, 21, 17, 13, 9, 5])
        self.assertEqual(RingPath(1, reverse=True).screens, [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45])
        self.assertEqual(RingPath(0).screens, [0, 40, 32, 24, 16, 8])
        self.assertEqual(RingPath(1).count, 12)
        self.assertEqual(RingPath(4).count, 6)

    def test_each_wall_gets_its_own_cells_with_their_place_round_the_ring(self):
        path = RingPath(1)       # screens (from 0) 1 | 45, 41, 37, 33, 29, 25 | 21, 17, 13, 9, 5: height 1, grid row 1
        self.assertEqual(path.cells(0), [(6, 0), (11, 7), (10, 8), (9, 9), (8, 10), (7, 11)])
        self.assertEqual(path.cells(1), [(11, 1), (10, 2), (9, 3), (8, 4), (7, 5), (6, 6)])
        self.assertEqual(sorted(place for wall in (0, 1) for _, place in path.cells(wall)), list(range(12)))


class TestLoops(unittest.TestCase):
    def test_each_row_gets_a_word(self):
        loops = plan_loops([1, 3], ["Matthias", "Oostrik"], False)
        self.assertEqual([(word, path.row) for word, path in loops], [("Matthias", 1), ("Oostrik", 3)])
        self.assertEqual([word for word, _ in plan_loops([0, 1, 2], ["One"], False)], ["One"] * 3)
        self.assertEqual(plan_loops([1], [], False), [])
        self.assertEqual(plan_loops([], ["A"], False), [])


class TestPosition(unittest.TestCase):
    def test_the_word_enters_the_first_screen_at_its_right_edge_and_moves_left(self):
        # before moving, the strip's x = 0 is at the screen's right edge: the screen shows nothing of it yet
        self.assertEqual(scroll_offset(0, 0.0, 0.0), -SCREEN_PX)
        # after a screen's width, the strip's start is at the screen's left edge
        self.assertEqual(scroll_offset(0, SCREEN_PX, 0.0), 0.0)
        # the next screen is one screen and a gap further along
        self.assertEqual(scroll_offset(1, SCREEN_PX, 100.0), -SCREEN_PX - 100.0)


class TestStrip(unittest.TestCase):
    def test_the_strip_is_the_screen_high_with_the_ink_the_size_in_the_middle(self):
        strip = name_strip("Matthias", 640, 512)
        self.assertEqual(strip.shape[0], 640)
        self.assertEqual(strip.shape[2], 4)
        alpha = strip[:, :, 3]
        rows = np.flatnonzero(alpha.max(axis=1))
        self.assertTrue(rows.size)
        self.assertGreaterEqual(rows[0], 64 - 1)
        self.assertLessEqual(rows[-1], 640 - 64)
        # the letters with their edge fill the size
        self.assertGreater(rows[-1] - rows[0], 0.9 * 512)
        # in the vertical middle
        self.assertLessEqual(abs(rows[0] - (639 - rows[-1])), 3)
        # white letters on a black edge: both colors are there where the alpha is
        self.assertTrue((strip[:, :, :3][alpha > 0] == 255).any())
        self.assertTrue((strip[:, :, :3][alpha > 0] == 0).any())
        # several screens long
        self.assertGreater(strip.shape[1], 3 * SCREEN_PX)

    def test_a_bigger_size_makes_a_longer_word_and_the_screen_is_the_most(self):
        self.assertGreater(name_strip("Matthias", 640, 640).shape[1], name_strip("Matthias", 640, 440).shape[1])
        self.assertEqual(name_strip("Matthias", 640, 900).shape[1], name_strip("Matthias", 640, 640).shape[1])

    def test_an_empty_word_is_an_empty_strip(self):
        self.assertEqual(name_strip("", 640, 512).shape, (640, 1, 4))


def settings(**values) -> SimpleNamespace:
    base = dict(credit_name="Matthias Oostrik", credit_scroll=True, credit_scroll_rows="1,3", credit_scroll_speed_1=1.0,
                credit_scroll_speed_2=0.5, credit_scroll_size=512, credit_scroll_gap=288, credit_scroll_reverse=False,
                credit_scroll_last='x')
    base.update(values)
    return SimpleNamespace(**base)


class TestScroller(unittest.TestCase):
    def setUp(self):
        self.sent: list[tuple] = []        # wall, word, speed, gap, size, ring, started, cells
        self.settings = settings()
        self.scroller = CreditScroller(self.settings, lambda *args: self.sent.append(args))

    def test_the_readout_starts_empty(self):
        self.assertEqual(self.settings.credit_scroll_last, '')

    def test_the_loops_go_out_every_second_with_the_time_they_began(self):
        self.scroller.update(100.0, False)
        self.scroller.update(100.5, False)
        self.assertEqual(len(self.sent), 4)        # two rows, both walls
        self.assertEqual({word for _, word, *_ in self.sent}, {"Matthias", "Oostrik"})
        for wall, word, speed, gap, size, ring, started, cells in self.sent:
            self.assertIn(wall, (0, 1))
            # each word at its own speed
            self.assertEqual((speed, gap, size, ring), (512.0 if word == "Matthias" else 256.0, 288.0, 512, 12))
            self.assertEqual(started, 100.0)                       # both began when the switch went on
            self.assertEqual(len(cells), 6)                        # this wall's half of the ring
            self.assertTrue({place for _, place in cells} < set(range(12)))     # with their places round it
            self.assertTrue(all(0 <= cell < 24 for cell, _ in cells))
        self.assertEqual(self.settings.credit_scroll_last, 'Matthias round row 1, Oostrik round row 3')
        self.sent.clear()
        self.scroller.update(101.0, False)
        self.assertEqual({started for *_, started, _ in self.sent}, {100.0})     # a resend moves nothing

    def test_a_changed_speed_leaves_the_word_where_it_is(self):
        self.scroller.update(0.0, False)
        self.settings.credit_scroll_speed_2 = 1.0           # Oostrik from 256 to 512 px/s after 10 s: 2560 px so far
        self.sent.clear()
        self.scroller.update(10.0, False)
        for _, word, speed, *_, started, _ in self.sent:
            if word == "Oostrik":
                self.assertEqual(speed, 512.0)
                self.assertEqual(started, 10.0 - 2560 / 512)        # 2560 px at the new speed: began 5 s ago
            else:
                self.assertEqual(started, 0.0)

    def test_nothing_goes_out_while_off_or_blocked_and_the_loops_begin_again_from_the_camera(self):
        self.settings.credit_scroll = False
        self.scroller.update(10.0, False)
        self.assertEqual(self.sent, [])
        self.settings.credit_scroll = True
        self.scroller.update(20.0, False)
        self.assertTrue(self.sent)
        self.sent.clear()
        self.scroller.update(30.0, True)
        self.assertEqual(self.sent, [])
        self.assertEqual(self.settings.credit_scroll_last, '')
        self.scroller.update(40.0, False)
        self.assertEqual({started for _, word, *_, started, _ in self.sent if word == "Matthias"}, {40.0})

    def test_no_valid_row_is_a_warning_not_a_loop(self):
        self.settings.credit_scroll_rows = "7"
        with self.assertLogs('apps.smile.credit_scroll', 'WARNING'):
            self.assertEqual(self.scroller.send_loops(0.0), [])
        self.assertEqual(self.sent, [])

    def test_a_wall_that_fails_does_not_stop_the_other(self):
        def send(wall, *args):
            if wall == 0:
                raise OSError('no wall')
            self.sent.append((wall, *args))
        scroller = CreditScroller(self.settings, send)
        with self.assertLogs('apps.smile.credit_scroll'):
            scroller.send_loops(0.0)
        self.assertEqual(len(self.sent), 2)
        self.assertTrue(all(wall == 1 for wall, *_ in self.sent))


if __name__ == '__main__':
    unittest.main()
