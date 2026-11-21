"""Tests for the OSC receiver: what a bound callback gets, a failing callback, a port change (loopback sockets)."""

import socket
import threading
import time
import unittest

from pythonosc.udp_client import SimpleUDPClient

from modules.inout import OscReceiver, OscReceiverSettings


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]


class TestReceiver(unittest.TestCase):
    def setUp(self):
        self.settings = OscReceiverSettings()
        self.settings.return_messages = False
        self.settings.port_in = free_port()
        self.receiver = OscReceiver(self.settings)
        self.received: list[tuple] = []
        self.got = threading.Event()
        self.client: SimpleUDPClient | None = None

    def tearDown(self):
        self.receiver.stop()
        if self.client is not None:
            self.client.close()

    def callback(self, *args) -> None:
        self.received.append(args)
        self.got.set()

    def send(self, address: str, *args) -> None:
        if self.client is not None:
            self.client.close()
        self.client = SimpleUDPClient('127.0.0.1', self.settings.port_in)
        self.client.send_message(address, list(args))

    def test_a_bound_callback_gets_its_arguments(self):
        self.receiver.bind('/level', self.callback)
        self.receiver.start()
        self.send('/level', 0.5, 3)
        self.assertTrue(self.got.wait(2.0))
        self.assertEqual(self.received, [(0.5, 3)])

    def test_a_failing_callback_does_not_stop_the_next(self):
        def failing(*args) -> None:
            raise ValueError('a broken callback')
        self.receiver.bind('/level', failing)
        self.receiver.bind('/level', self.callback)
        self.receiver.start()
        with self.assertLogs('modules.utils.broadcast', 'ERROR'):
            self.send('/level', 1.0)
            self.assertTrue(self.got.wait(2.0))
        self.assertEqual(self.received, [(1.0,)])

    def test_a_port_change_keeps_the_bindings(self):
        self.receiver.bind('/level', self.callback)
        self.receiver.start()
        self.settings.port_in = free_port()
        end = time.perf_counter() + 2.0
        while not self.got.is_set() and time.perf_counter() < end:     # until the new server takes it
            self.send('/level', 0.25)
            self.got.wait(0.05)
        self.assertTrue(self.got.is_set())
        self.assertEqual(self.received[0], (0.25,))

    def test_a_failed_echo_is_logged_and_the_callbacks_still_run(self):
        self.settings.return_messages = True
        self.settings.ip_out = '255.255.255.255'            # a broadcast address: a socket without SO_BROADCAST refuses
        self.receiver.bind('/level', self.callback)
        self.receiver.start()
        with self.assertLogs('modules.inout.osc_receiver', 'ERROR'):
            self.send('/level', 0.5)
            self.assertTrue(self.got.wait(2.0))
            time.sleep(0.05)                                # the echo comes after the callbacks
        self.assertEqual(self.received, [(0.5,)])

    def test_an_invalid_return_address_keeps_the_current_one(self):
        self.receiver.start()
        client = self.receiver._return_client
        with self.assertLogs('modules.inout.osc_receiver', 'WARNING'):
            self.settings.ip_out = 'not an address'
        self.assertIs(self.receiver._return_client, client)

    def test_it_binds_its_settings_at_start_and_unbinds_them_at_stop(self):
        self.receiver.start()
        self.receiver.stop()
        self.settings.port_in = free_port()                 # after stop: no callback restarts the server
        time.sleep(0.05)
        self.assertIsNone(self.receiver._server)


if __name__ == '__main__':
    unittest.main()
