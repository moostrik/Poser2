"""Tests for the Art-Net sender: its ArtDmx bytes, what goes out on a change and as a keep-alive, its rate, its
blackouts, two universes and a failed send (a fake socket, no network)."""

import time
import unittest
from unittest import mock

from modules.inout import ArtNetSender, ArtNetSenderSettings
from modules.inout.artnet_sender import ARTNET_PORT, KEEP_ALIVE, MAX_FPS, artdmx_header

FLASH = 'flash'
BLACK = 'black'


class FakeSocket:
    """Records per universe what each ArtDmx packet carried; `fail` sends raise."""

    def __init__(self) -> None:
        self.sent: dict[int, list] = {}
        self.targets: set[tuple[str, int]] = set()
        self.fail: bool = False

    def sendto(self, datagram: bytes, target: tuple[str, int]) -> int:
        if self.fail:
            raise OSError('network is unreachable')
        self.targets.add(target)
        universe: int = datagram[14] | datagram[15] << 8
        data: bytes = datagram[18:]
        self.sent.setdefault(universe, []).append(
            FLASH if data == b'\xff' * len(data) else BLACK if data == bytes(len(data)) else data)
        return len(datagram)

    def close(self) -> None:
        pass


class TestArtDmx(unittest.TestCase):
    def test_the_header_is_art_net_4_s(self):
        # the ID, opcode 0x5000 low byte first, version 14, sequence and physical 0, universe low first, length high first
        self.assertEqual(artdmx_header(1, 4), b'Art-Net\x00\x00\x50\x00\x0e\x00\x00\x01\x00\x00\x04')
        self.assertEqual(artdmx_header(254, 512), b'Art-Net\x00\x00\x50\x00\x0e\x00\x00\xfe\x00\x02\x00')


class TestArtNetSender(unittest.TestCase):
    def setUp(self):
        self.socket = FakeSocket()
        self.patch = mock.patch('modules.inout.artnet_sender.socket')
        self.socket_module = self.patch.start()               # the socket module as the sender sees it
        self.socket_module.socket.return_value = self.socket
        self.settings = ArtNetSenderSettings()
        self.value = 7

    def tearDown(self):
        self.patch.stop()

    def fill(self, packet: bytearray) -> None:
        packet[0] = self.value

    def opened(self, universes: int = 1, flash: bool = True) -> ArtNetSender:
        """A sender with its socket, not started: its ticks are the test's."""
        sender = ArtNetSender(self.settings, 4, self.fill, universes, flash)
        sender._socket = self.socket_module.socket()
        sender._update_target()
        return sender

    def test_a_failed_send_is_logged_and_the_next_goes_out(self):
        sender = self.opened()
        self.socket.fail = True
        with self.assertLogs('modules.inout.artnet_sender', 'ERROR'):
            sender.update(0.0)
        self.socket.fail = False
        self.value = 8
        sender.update(0.02)
        self.assertEqual(self.socket.sent[0], [bytes([8, 0, 0, 0])])
        self.assertEqual(self.socket.targets, {('127.0.0.1', ARTNET_PORT)})

    def test_a_universe_goes_out_on_a_change(self):
        sender = self.opened()
        sender.update(0.0)
        sender.update(0.02)
        self.value = 8
        sender.update(0.04)
        self.assertEqual(self.socket.sent[0], [bytes([7, 0, 0, 0]), bytes([8, 0, 0, 0])])

    def test_an_unchanged_universe_goes_out_as_a_keep_alive(self):
        sender = self.opened()
        sender.update(0.0)
        sender.update(KEEP_ALIVE - 0.01)
        sender.update(KEEP_ALIVE)
        self.assertEqual(self.socket.sent[0], [bytes([7, 0, 0, 0])] * 2)

    def test_the_rate_is_at_most_44_hz(self):
        self.settings.fps = 120.0
        sender = self.opened()
        self.assertGreaterEqual(sender._interval(), 1.0 / MAX_FPS)

    def test_two_universes_each_send_their_part(self):
        def fill(packet: bytearray) -> None:
            packet[0] = 1
            packet[4] = 2
        self.settings.universe = 3
        sender = ArtNetSender(self.settings, 4, fill, universes=2)
        sender._socket = self.socket_module.socket()
        sender._update_target()
        sender.update(0.0)
        self.assertEqual(self.socket.sent, {3: [bytes([1, 0, 0, 0])], 4: [bytes([2, 0, 0, 0])]})

    def test_start_switching_off_and_stop_black_out_after_the_flash(self):
        self.settings.fps = MAX_FPS
        sender = ArtNetSender(self.settings, 4, self.fill)
        sender.start()
        time.sleep(0.15)
        self.settings.enabled = False
        time.sleep(0.15)
        sender.stop()
        sent = self.socket.sent[0]
        self.assertEqual(sent[:3], [FLASH, BLACK, bytes([7, 0, 0, 0])])      # start, then the packet
        self.assertEqual(sent[-4:], [FLASH, BLACK, FLASH, BLACK])               # switched off: nothing more; stop

    def test_without_the_flash_only_the_blackouts(self):
        self.settings.fps = MAX_FPS
        sender = ArtNetSender(self.settings, 4, self.fill, flash=False)
        sender.start()
        time.sleep(0.1)
        self.settings.enabled = False
        time.sleep(0.1)
        sender.stop()
        self.assertEqual(self.socket.sent[0], [BLACK, bytes([7, 0, 0, 0]), BLACK, BLACK])

    def test_switched_on_it_flashes_and_sends_the_packet_again(self):
        sender = self.opened()
        sender.update(0.0)
        sender._on_enabled(True)
        sender.update(0.02)
        sender.update(0.04)
        self.assertEqual(self.socket.sent[0], [bytes([7, 0, 0, 0]), FLASH, bytes([7, 0, 0, 0])])

    def test_the_packet_size_is_even(self):
        # an ArtDmx packet's data is an even number of channels
        for channels in (9, 10, 82, 83):
            sender = ArtNetSender(ArtNetSenderSettings(), channels, lambda packet: None)
            self.assertEqual(sender.packet_size % 2, 0)
            self.assertGreaterEqual(sender.packet_size, channels)
            self.assertEqual(len(sender.packet), sender.packet_size)


if __name__ == '__main__':
    unittest.main()
