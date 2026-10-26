"""Tests for the waiting line and what each phone is told."""

import unittest

from relay.line import ASLEEP, BLOCKS, LOOKING, OUTSIDE, Line


class TestLine(unittest.TestCase):
    def setUp(self):
        self.line = Line()
        self.line.set_eye(True, free=True, status=LOOKING)

    def test_joining_gives_a_place_and_joining_again_keeps_it(self):
        self.assertEqual(self.line.join('a'), 1)
        self.assertEqual(self.line.join('b'), 2)
        self.assertEqual(self.line.join('a'), 1)
        self.assertEqual(self.line.waiting, ['a', 'b'])

    def test_the_first_in_line_gets_the_turn(self):
        self.line.join('a')
        self.line.join('b')
        self.assertEqual(self.line.start_turn(), 'a')
        self.assertEqual(self.line.waiting, ['b'])
        self.assertEqual(self.line.start_turn(), 'a')      # asked again while it runs: the same turn
        self.assertEqual(self.line.end_turn(), 'a')
        self.assertIsNone(self.line.turn)
        self.assertEqual(self.line.start_turn(), 'b')

    def test_nobody_waiting_is_no_turn(self):
        self.assertIsNone(self.line.start_turn())
        self.assertIsNone(self.line.end_turn())

    def test_leaving_the_turn_says_so(self):
        self.line.join('a')
        self.line.start_turn()
        self.assertTrue(self.line.leave('a'))
        self.assertIsNone(self.line.turn)
        self.assertFalse(self.line.leave('a'))

    def test_the_phone_with_the_turn_does_not_rejoin_the_line(self):
        self.line.join('a')
        self.line.start_turn()
        self.assertEqual(self.line.join('a'), 0)
        self.assertEqual(self.line.waiting, [])


class TestState(unittest.TestCase):
    def setUp(self):
        self.line = Line()

    def test_asleep_without_smile(self):
        state = self.line.state('a')
        self.assertEqual((state['eye'], state['status'], state['block']), ('asleep', ASLEEP, BLOCKS[0]))

    def test_free_and_busy(self):
        self.line.set_eye(True, free=True, status=LOOKING)
        self.assertEqual(self.line.state('a')['eye'], 'free')
        self.line.set_eye(True, free=False, status='looking at someone inside')
        state = self.line.state('a')
        self.assertEqual((state['eye'], state['status']), ('busy', 'looking at someone inside'))

    def test_waiting_counts_down_in_blocks(self):
        self.line.set_eye(True, free=True, status=LOOKING)
        for phone in 'abcde':
            self.line.join(phone)
        self.assertEqual(self.line.state('a')['block'], BLOCKS[-1])
        self.assertEqual(self.line.state('d')['block'], BLOCKS[0])
        self.assertEqual(self.line.state('e')['block'], BLOCKS[0])
        self.assertEqual(self.line.state('b')['status'], f'1 before you, {LOOKING}')
        self.assertEqual(self.line.state('a')['status'], LOOKING)

    def test_a_turn_is_you_and_outside_for_the_others(self):
        self.line.set_eye(True, free=True, status=LOOKING)
        self.line.join('a')
        self.line.join('b')
        self.line.start_turn()
        self.assertEqual(self.line.state('a')['eye'], 'you')
        self.assertEqual(self.line.state('a')['block'], 1)
        self.assertEqual(self.line.state('b')['status'], OUTSIDE)
        self.assertEqual(self.line.state('c')['eye'], 'busy')

    def test_smile_leaving_ends_the_turn(self):
        self.line.set_eye(True, free=True)
        self.line.join('a')
        self.line.start_turn()
        self.line.set_eye(False)
        self.assertIsNone(self.line.turn)
        self.assertEqual(self.line.state('a')['eye'], 'asleep')


if __name__ == '__main__':
    unittest.main()
