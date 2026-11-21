"""Tests for the OSC sender: what goes into a bundle, a send that fails, and its timing (a loopback socket)."""

import socket
import time
import unittest

from pythonosc.osc_bundle import OscBundle

from modules.inout import OscSender, OscSenderSettings
from modules.inout.osc_sender import _tick_messages

UNSENDABLE_IP: str = '255.255.255.255'      # a broadcast address: a socket without SO_BROADCAST refuses the send


class Listener:
    """A loopback UDP socket the sender sends to."""

    def __init__(self) -> None:
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(('127.0.0.1', 0))
        self.socket.settimeout(2.0)
        self.port: int = self.socket.getsockname()[1]

    def receive(self) -> tuple[float, list[tuple[str, list]]]:
        """When the next bundle came in, and its messages."""
        datagram: bytes = self.socket.recv(65536)
        received: float = time.perf_counter()
        return received, [(m.address, m.params) for m in OscBundle(datagram)]

    def close(self) -> None:
        self.socket.close()


class TestSender(unittest.TestCase):
    def setUp(self):
        self.listener = Listener()
        self.settings = OscSenderSettings()
        self.settings.fps = 120.0
        self.settings.port = self.listener.port
        self.sender = OscSender(self.settings, events=('/bang', '/test_bang'))

    def tearDown(self):
        self.sender.stop()
        self.listener.close()

    def test_a_state_sends_its_newest_value_and_an_event_every_one(self):
        data = [('/progress', 0.1), ('/bang', 0), ('/progress', 0.2), ('/bang', 2), ('/state', 'CAMERA_IN'),
                ('/test_bang', (0, 1)), ('/test_bang', (1, 1))]
        self.assertEqual(_tick_messages(data, self.sender.events),
                         [('/progress', 0.2), ('/bang', 0), ('/bang', 2), ('/state', 'CAMERA_IN'),
                          ('/test_bang', (0, 1)), ('/test_bang', (1, 1))])

    def test_a_tuple_goes_as_separate_arguments(self):
        self.sender.start()
        self.sender.submit('/test_bang', (5, 1))
        _, messages = self.listener.receive()
        self.assertEqual(messages, [('/test_bang', [5, 1])])

    def test_nothing_is_queued_while_the_sender_does_not_run(self):
        self.sender.submit('/progress', 0.5)
        self.assertTrue(self.sender._queue.empty())

    def test_a_failed_send_does_not_stop_the_sender(self):
        self.settings.ip = UNSENDABLE_IP
        self.sender.start()
        with self.assertLogs('modules.inout.osc_client', 'ERROR'):
            self.sender.submit('/progress', 0.1)
            end = time.perf_counter() + 2.0
            while not self.sender._queue.empty() and time.perf_counter() < end:
                time.sleep(0.01)
            time.sleep(0.05)
        self.settings.ip = '127.0.0.1'
        self.sender.submit('/progress', 0.2)
        _, messages = self.listener.receive()
        self.assertEqual([address for address, _ in messages], ['/progress'])
        self.assertAlmostEqual(messages[0][1][0], 0.2, places=6)
        self.assertTrue(self.sender.running)

    def test_an_event_goes_out_at_once(self):
        self.settings.fps = 1.0
        self.sender.start()
        time.sleep(0.05)                                    # past the first tick
        submitted = time.perf_counter()
        self.sender.submit('/bang', 1)
        received, messages = self.listener.receive()
        self.assertEqual(messages, [('/bang', [1])])
        self.assertLess(received - submitted, 0.1)

    def test_stop_ends_the_wait_at_once(self):
        self.settings.fps = 1.0
        self.sender.start()
        time.sleep(0.05)
        start = time.perf_counter()
        self.sender.stop()
        self.assertLess(time.perf_counter() - start, 0.1)

    def test_its_waker_lives_from_start_to_stop(self):
        self.assertIsNone(self.sender._waker)               # a sender that never runs holds no handles
        self.sender.start()
        self.assertIsNotNone(self.sender._waker)
        self.sender.stop()
        self.assertIsNone(self.sender._waker)

    def test_it_binds_the_address_at_start_and_unbinds_it_at_stop(self):
        self.sender.start()
        self.sender.stop()
        self.settings.ip = UNSENDABLE_IP                    # after stop: no callback rebuilds the client
        self.assertEqual(self.sender._client._target, ('127.0.0.1', self.listener.port))


if __name__ == '__main__':
    unittest.main()
