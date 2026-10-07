"""Tests for the OSC receiver: real UDP sockets on localhost, on ports the system picks."""

import threading
import time
import unittest

from pythonosc.udp_client import SimpleUDPClient

from modules.osc.Receiver import Receiver

TIMEOUT: float = 2.0


class TestReceiver(unittest.TestCase):
    def setUp(self):
        self.got: list[tuple] = []
        self.arrived = threading.Event()

        def on_light(*args) -> None:
            self.got.append(args)
            self.arrived.set()

        self.receiver = Receiver(0, {'/light': on_light})
        self.receiver.start()
        self.addCleanup(self.receiver.stop)
        self.assertTrue(self.receiver.running)
        assert self.receiver.address is not None
        self.port: int = self.receiver.address[1]
        self.client = SimpleUDPClient('127.0.0.1', self.port)

    def test_a_message_reaches_its_handler_with_its_arguments(self):
        self.client.send_message('/light', [3, 1.5])
        self.assertTrue(self.arrived.wait(TIMEOUT), 'no message arrived')
        self.assertEqual(self.got, [(3, 1.5)])

    def test_a_handler_that_raises_is_logged_and_the_next_message_still_arrives(self):
        calls: list[tuple] = []
        second = threading.Event()

        def bad(*args) -> None:
            calls.append(args)
            if len(calls) == 2:
                second.set()
            raise ValueError('bad cell')

        receiver = Receiver(0, {'/light': bad})
        receiver.start()
        self.addCleanup(receiver.stop)
        assert receiver.address is not None
        client = SimpleUDPClient('127.0.0.1', receiver.address[1])
        # the second message is handled after the first one's failure was logged (one request at a time)
        with self.assertLogs('modules.osc.Receiver', 'ERROR'):
            client.send_message('/light', [1, 1.0])
            client.send_message('/light', [2, 1.0])
            self.assertTrue(second.wait(TIMEOUT), 'the second message did not arrive')
        self.assertEqual(calls, [(1, 1.0), (2, 1.0)])

    def test_a_port_in_use_is_a_warning_and_stop_may_run_twice(self):
        other = Receiver(self.port, {'/light': lambda *args: None})
        with self.assertLogs('modules.osc.Receiver', 'WARNING'):
            other.start()
        self.assertFalse(other.running)
        self.assertIsNone(other.address)
        other.stop()
        other.stop()
        self.receiver.stop()
        self.receiver.stop()
        self.assertFalse(self.receiver.running)
        # the port is free again
        again = Receiver(self.port, {'/light': lambda *args: None})
        again.start()
        self.addCleanup(again.stop)
        self.assertTrue(again.running)


if __name__ == '__main__':
    unittest.main()
