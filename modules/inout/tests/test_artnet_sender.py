"""Tests for the Art-Net sender's packet, flash and stop (a fake StupidArtnet, no network)."""

import time
import unittest
from unittest.mock import Mock

from stupidArtnet import StupidArtnet

from modules.inout import ArtNetSender, ArtNetSenderSettings


def fake_artnet() -> Mock:
    """A StupidArtnet whose calls are recorded, in order, in `method_calls`."""
    return Mock(spec=StupidArtnet)


def calls(fake: Mock) -> list[str]:
    return [name for name, _args, _kwargs in fake.method_calls]


class TestArtNetSender(unittest.TestCase):
    def test_a_flash_goes_out_on_the_next_tick_in_place_of_the_packet(self):
        sender = ArtNetSender(ArtNetSenderSettings(), 4, lambda packet: None)
        fake = fake_artnet()
        sender.sender = fake
        sender.flash()
        self.assertEqual(calls(fake), [])
        sender.update()
        sender.update()
        self.assertEqual(calls(fake), ['flash_all', 'send'])

    def test_stop_ends_with_a_flash_then_a_blackout(self):
        settings = ArtNetSenderSettings()
        settings.fps = 50.0
        sender = ArtNetSender(settings, 4, lambda packet: None)
        sender.start()
        fake = fake_artnet()
        with sender.mutex:
            sender.sender = fake
        time.sleep(0.1)
        start = time.perf_counter()
        sender.stop()
        self.assertLess(time.perf_counter() - start, 0.5)
        self.assertFalse(sender.is_alive())
        self.assertEqual(calls(fake)[-2:], ['flash_all', 'blackout'])
        self.assertIn('send', calls(fake))

    def test_the_packet_size_is_even(self):
        # StupidArtnet rounds its own size up to even and refuses a packet of another length
        for channels in (9, 10, 82, 83):
            sender = ArtNetSender(ArtNetSenderSettings(), channels, lambda packet: None)
            self.assertEqual(sender.packet_size % 2, 0)
            self.assertGreaterEqual(sender.packet_size, channels)
            self.assertEqual(len(sender.packet), sender.packet_size)

    def test_fill_writes_the_packet_that_is_sent(self):
        def fill(packet: bytearray) -> None:
            packet[0] = 7
        sender = ArtNetSender(ArtNetSenderSettings(), 4, fill)
        sender.update()
        self.assertEqual(sender.packet[0], 7)


if __name__ == '__main__':
    unittest.main()
