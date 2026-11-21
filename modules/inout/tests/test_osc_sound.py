"""Tests for the sound sender's sending: a failed send, the switch, and its address (a loopback socket)."""

import socket
import time
import unittest

from pythonosc.osc_bundle import OscBundle

from modules.inout import OscSound, OscSoundSettings
from modules.pose import Frame

UNSENDABLE_IP: str = '255.255.255.255'      # a broadcast address: a socket without SO_BROADCAST refuses the send


class TestOscSound(unittest.TestCase):
    def setUp(self):
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.listener.bind(('127.0.0.1', 0))
        self.listener.settimeout(2.0)
        self.settings = OscSoundSettings()
        self.settings.port = self.listener.getsockname()[1]
        self.sound = OscSound(self.settings)

    def tearDown(self):
        self.sound.stop()
        self.listener.close()

    def receive(self) -> dict[str, list]:
        return {m.address: m.params for m in OscBundle(self.listener.recv(65536))}

    def submit(self) -> None:
        self.sound.submit_frames(self.settings.stage, {0: Frame(0, 0, 0.0)})

    def test_a_bundle_goes_out_for_each_submit(self):
        self.sound.start()
        self.submit()
        self.assertEqual(self.receive()['/pose/0/active'], [1])

    def test_a_failed_send_does_not_stop_it(self):
        self.settings.ip = UNSENDABLE_IP
        self.sound.start()
        with self.assertLogs('modules.inout.osc_client', 'ERROR'):
            self.submit()
            time.sleep(0.1)
        self.settings.ip = '127.0.0.1'
        self.submit()
        self.assertEqual(self.receive()['/pose/0/active'], [1])
        self.assertTrue(self.sound.running)

    def test_switched_off_it_sends_one_blackout_and_then_nothing(self):
        self.sound.start()
        self.settings.enabled = False
        self.submit()
        self.assertEqual(self.receive()['/pose/0/active'], [0])
        self.submit()
        self.listener.settimeout(0.2)
        with self.assertRaises(socket.timeout):
            self.listener.recv(65536)
        self.sound.stop()                               # the blackout went out already
        with self.assertRaises(socket.timeout):
            self.listener.recv(65536)

    def test_stop_sends_a_blackout(self):
        self.sound.start()
        self.sound.stop()
        self.assertEqual(self.receive()['/global/state'], [-1])

    def test_it_binds_the_address_at_start_and_unbinds_it_at_stop(self):
        port: int = self.settings.port
        self.sound.start()
        self.sound.stop()
        self.receive()                                  # the blackout
        self.settings.ip = UNSENDABLE_IP                # after stop: no callback rebuilds the client
        self.assertEqual(self.sound._client._target, ('127.0.0.1', port))


if __name__ == '__main__':
    unittest.main()
