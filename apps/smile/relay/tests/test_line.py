"""Tests for the waiting line and what each phone is told."""

import unittest

from apps.smile.relay.line import ASLEEP, BLOCKS, BUSY, INSIDE, LEAST_LEFT, LOOKING, OUTSIDE, TYPICAL_TURN, WARMING, Line


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
        self.assertEqual((self.line.state('a')['reason'], self.line.state('a')['eta']), ('asleep', None))


class TestWait(unittest.TestCase):
    """The reason a phone waits and how long: what runs now has its rest, every phone ahead a typical turn."""
    def setUp(self):
        self.line = Line()
        self.line.set_eye(True, free=True, status=LOOKING, now=100.0)
        for phone in 'abc':
            self.line.join(phone)

    def wait(self, phone: str, now: float) -> tuple[str, float | None]:
        state = self.line.state(phone, now)
        return state['reason'], state['eta']

    def test_free_is_the_least_wait_and_a_turn_per_phone_ahead(self):
        self.assertEqual(self.wait('a', 100.0), ('free', LEAST_LEFT))
        self.assertEqual(self.wait('c', 100.0), ('free', LEAST_LEFT + 2 * TYPICAL_TURN))
        self.assertEqual(self.wait('x', 100.0), ('free', LEAST_LEFT + 3 * TYPICAL_TURN))      # not in line: behind everyone

    def test_someone_inside_has_the_rest_of_a_typical_turn(self):
        self.line.set_eye(True, free=False, status=INSIDE, now=100.0)
        self.assertEqual(self.wait('a', 110.0), ('inside', TYPICAL_TURN - 10.0))
        self.assertEqual(self.wait('b', 110.0), ('inside', 2 * TYPICAL_TURN - 10.0))
        self.assertEqual(self.wait('a', 300.0), ('inside', LEAST_LEFT))                      # never already over
        self.line.set_eye(True, free=False, status=INSIDE, now=300.0)                        # the same status: the start stays
        self.assertEqual(self.wait('a', 300.0), ('inside', LEAST_LEFT))

    def test_an_online_turn_counts_down_and_is_timed(self):
        self.line.start_turn(100.0)
        self.assertEqual(self.wait('b', 120.0), ('outside', TYPICAL_TURN - 20.0))
        self.line.end_turn(145.0)
        self.assertEqual(list(self.line.durations), [45.0])
        self.assertEqual(self.line.typical(), 45.0)
        self.line.start_turn(200.0)
        self.assertEqual(self.wait('c', 200.0), ('outside', 45.0))

    def test_the_typical_turn_is_the_median_of_the_last_ones(self):
        self.line.join('d')
        for duration in (30.0, 50.0, 40.0, 400.0):      # four phones, four turns; the median shrugs at the long one
            self.assertIsNotNone(self.line.start_turn(0.0))
            self.line.end_turn(duration)
        self.assertEqual(self.line.typical(), 45.0)

    def test_warming_up_and_busy_give_no_estimate(self):
        self.line.set_eye(True, free=False, status=WARMING, now=100.0)
        self.assertEqual(self.wait('a', 100.0), ('warming', None))
        self.line.set_eye(True, free=False, status=BUSY, now=100.0)
        self.assertEqual(self.wait('a', 100.0), ('busy', None))


if __name__ == '__main__':
    unittest.main()
