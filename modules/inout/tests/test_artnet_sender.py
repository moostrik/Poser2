"""Tests for the Art-Net sender: what goes out on a change and as a keep-alive, its rate, its blackouts and two
universes (fake StupidArtnets, no network)."""

import time
import unittest
from unittest import mock

from stupidArtnet import StupidArtnet

from modules.inout import ArtNetSender, ArtNetSenderSettings
from modules.inout.artnet_sender import KEEP_ALIVE, MAX_FPS

FLASH = 'flash'
BLACK = 'black'


class FakeArtnets:
    """Replaces StupidArtnet in the sender's module; records per universe what each show sent."""

    def __init__(self) -> None:
        self.sent: dict[int, list] = {}
        self.patch = mock.patch('modules.inout.artnet_sender.StupidArtnet', side_effect=self._make)

    def _make(self, target_ip: str, universe: int, packet_size: int, fps: int) -> mock.Mock:
        fake = mock.Mock(spec=StupidArtnet)
        sent: list = self.sent.setdefault(universe, [])
        buffer: list[bytes] = [b'']

        def set_buffer(data: bytearray) -> None:
            buffer[0] = bytes(data)

        def show() -> None:
            data = buffer[0]
            sent.append(FLASH if data == b'\xff' * len(data) else BLACK if data == bytes(len(data)) else data)
        fake.set.side_effect = set_buffer
        fake.show.side_effect = show
        return fake


class TestArtNetSender(unittest.TestCase):
    def setUp(self):
        self.fakes = FakeArtnets()
        self.fakes.patch.start()
        self.settings = ArtNetSenderSettings()
        self.value = 7

    def tearDown(self):
        self.fakes.patch.stop()

    def fill(self, packet: bytearray) -> None:
        packet[0] = self.value

    def opened(self, universes: int = 1, flash: bool = True) -> ArtNetSender:
        """A sender with its StupidArtnets, not started: its ticks are the test's."""
        sender = ArtNetSender(self.settings, 4, self.fill, universes, flash)
        sender._update_senders()
        return sender

    def test_a_universe_goes_out_on_a_change(self):
        sender = self.opened()
        sender.update(0.0)
        sender.update(0.02)
        self.value = 8
        sender.update(0.04)
        self.assertEqual(self.fakes.sent[0], [bytes([7, 0, 0, 0]), bytes([8, 0, 0, 0])])

    def test_an_unchanged_universe_goes_out_as_a_keep_alive(self):
        sender = self.opened()
        sender.update(0.0)
        sender.update(KEEP_ALIVE - 0.01)
        sender.update(KEEP_ALIVE)
        self.assertEqual(self.fakes.sent[0], [bytes([7, 0, 0, 0])] * 2)

    def test_the_rate_is_at_most_44_hz(self):
        self.settings.fps = 120.0
        sender = self.opened()
        self.assertGreaterEqual(sender._interval(), 1.0 / MAX_FPS)

    def test_two_universes_each_send_their_part(self):
        def fill(packet: bytearray) -> None:
            packet[0] = 1
            packet[4] = 2
        sender = ArtNetSender(self.settings, 4, fill, universes=2)
        self.settings.universe = 3
        sender._update_senders()
        sender.update(0.0)
        self.assertEqual(self.fakes.sent, {3: [bytes([1, 0, 0, 0])], 4: [bytes([2, 0, 0, 0])]})

    def test_start_switching_off_and_stop_black_out_after_the_flash(self):
        self.settings.fps = MAX_FPS
        sender = ArtNetSender(self.settings, 4, self.fill)
        sender.start()
        time.sleep(0.15)
        self.settings.enabled = False
        time.sleep(0.15)
        sender.stop()
        sent = self.fakes.sent[0]
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
        self.assertEqual(self.fakes.sent[0], [BLACK, bytes([7, 0, 0, 0]), BLACK, BLACK])

    def test_switched_on_it_flashes_and_sends_the_packet_again(self):
        sender = self.opened()
        sender.update(0.0)
        sender._on_enabled(True)
        sender.update(0.02)
        sender.update(0.04)
        self.assertEqual(self.fakes.sent[0], [bytes([7, 0, 0, 0]), FLASH, bytes([7, 0, 0, 0])])

    def test_the_packet_size_is_even(self):
        # StupidArtnet rounds its own size up to even and refuses a packet of another length
        for channels in (9, 10, 82, 83):
            sender = ArtNetSender(ArtNetSenderSettings(), channels, lambda packet: None)
            self.assertEqual(sender.packet_size % 2, 0)
            self.assertGreaterEqual(sender.packet_size, channels)
            self.assertEqual(len(sender.packet), sender.packet_size)


if __name__ == '__main__':
    unittest.main()
