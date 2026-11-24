"""Broadcast: order, a failing callback, remove, clear and bool."""

import unittest

from modules.utils import Broadcast


class BroadcastTest(unittest.TestCase):
    def test_calls_in_registration_order(self) -> None:
        heard: list[tuple[str, int]] = []
        broadcast: Broadcast[[int]] = Broadcast(name='test')
        broadcast.add_callback(lambda value: heard.append(('a', value)))
        broadcast.add_callback(lambda value: heard.append(('b', value)))
        broadcast(7)
        self.assertEqual(heard, [('a', 7), ('b', 7)])

    def test_a_second_add_keeps_the_first_place(self) -> None:
        heard: list[str] = []
        first = lambda: heard.append('first')
        second = lambda: heard.append('second')
        broadcast: Broadcast[[]] = Broadcast(name='test')
        broadcast.add_callback(first)
        broadcast.add_callback(second)
        broadcast.add_callback(first)
        broadcast()
        self.assertEqual(heard, ['first', 'second'])

    def test_a_failing_callback_is_logged_and_the_next_runs(self) -> None:
        heard: list[int] = []

        def failing(_value: int) -> None:
            raise RuntimeError('fails')

        broadcast: Broadcast[[int]] = Broadcast(name='test')
        broadcast.add_callback(failing)
        broadcast.add_callback(heard.append)
        with self.assertLogs('modules.utils.broadcast', level='ERROR') as logs:
            broadcast(3)
        self.assertEqual(heard, [3])
        self.assertIn('Error in a test callback', logs.output[0])

    def test_remove(self) -> None:
        heard: list[int] = []
        broadcast: Broadcast[[int]] = Broadcast(name='test')
        broadcast.add_callback(heard.append)
        broadcast.remove_callback(heard.append)
        broadcast.remove_callback(heard.append)     # not registered: nothing happens
        broadcast(1)
        self.assertEqual(heard, [])

    def test_clear(self) -> None:
        heard: list[int] = []
        broadcast: Broadcast[[int]] = Broadcast(name='test')
        broadcast.add_callback(heard.append)
        broadcast.add_callback(lambda value: heard.append(-value))
        broadcast.clear()
        broadcast(1)
        self.assertEqual(heard, [])

    def test_bool_says_whether_a_callback_is_registered(self) -> None:
        broadcast: Broadcast[[int]] = Broadcast(name='test')
        self.assertFalse(broadcast)
        callback = lambda _value: None
        broadcast.add_callback(callback)
        self.assertTrue(broadcast)
        broadcast.remove_callback(callback)
        self.assertFalse(broadcast)


if __name__ == '__main__':
    unittest.main()
