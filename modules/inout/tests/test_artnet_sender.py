"""Tests for the Art-Net sender's packet (no network: the sender is never started)."""

import unittest

from modules.inout import ArtNetSender, ArtNetSenderSettings


class TestArtNetSender(unittest.TestCase):
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
