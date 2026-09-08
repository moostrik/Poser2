"""Tests for the OSC wire contract with the fixture firmware.

The firmware (`apps/white_space/data/firmware/firmware.ino`) parses these datagrams at
fixed byte offsets and reads a fixed body length, so the chunking, the address spelling and the
send order are all load-bearing. These tests pin them down.
"""

import unittest

import numpy as np

from apps.white_space.inout.osc_light_sender import (
    FIRMWARE_CHUNK_SIZE, FIRMWARE_NUM_CHUNKS, OscLightSender, OscLightSenderSettings,
)
from apps.white_space.light import Frame, Tick

# The firmware reads a fixed-size OSC preamble before the pixel body: 12-byte padded address,
# 4-byte typetag, 4-byte blob length.
OSC_PREAMBLE = 20
RESOLUTION = 3600


def _frame() -> Frame:
    frame = Frame(RESOLUTION, Tick(0.0, 0.0, 120.0, 0.0, 0))
    frame.white = np.linspace(0.0, 1.0, RESOLUTION, dtype=np.float32)
    frame.blue = np.linspace(1.0, 0.0, RESOLUTION, dtype=np.float32)
    return frame


class ChunkCalculationTest(unittest.TestCase):
    def test_installation_config_matches_firmware(self) -> None:
        """3600 px over a 1500-byte MTU must yield exactly what the firmware hard-codes."""
        self.assertEqual(
            OscLightSender._calculate_optimal_chunks(RESOLUTION, 1500),
            (FIRMWARE_CHUNK_SIZE, FIRMWARE_NUM_CHUNKS),
        )


class ChunkMessageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = OscLightSenderSettings()
        self.messages = OscLightSender._build_chunk_messages(
            _frame(), self.settings, FIRMWARE_CHUNK_SIZE, FIRMWARE_NUM_CHUNKS
        )
        assert self.messages is not None
        self.addresses = [m.address for m in self.messages]

    def test_pixel_burst_is_six_datagrams(self) -> None:
        self.assertEqual(len(self.messages), 2 * FIRMWARE_NUM_CHUNKS)

    def test_channels_are_grouped_and_blue_last(self) -> None:
        """The firmware commits the frame on `/WS/blue2`, so it must be sent last."""
        self.assertEqual(
            self.addresses,
            ["/WS/white0", "/WS/white1", "/WS/white2", "/WS/blue0", "/WS/blue1", "/WS/blue2"],
        )

    def test_datagram_size_matches_firmware_read(self) -> None:
        """20 header bytes + a 1200-byte body is exactly what the firmware consumes."""
        for message in self.messages:
            self.assertEqual(len(message.dgram), OSC_PREAMBLE + FIRMWARE_CHUNK_SIZE, message.address)

    def test_chunk_digit_sits_where_the_firmware_looks(self) -> None:
        """`hdr[9]` for white, `hdr[8]` for blue — the firmware's fixed parse offsets."""
        for i in range(FIRMWARE_NUM_CHUNKS):
            white = self.messages[i].dgram
            blue = self.messages[FIRMWARE_NUM_CHUNKS + i].dgram
            self.assertEqual(white[1:2], b"W")
            self.assertEqual(white[4:5], b"w")
            self.assertEqual(white[9:10], str(i).encode())
            self.assertEqual(blue[1:2], b"W")
            self.assertEqual(blue[4:5], b"b")
            self.assertEqual(blue[8:9], str(i).encode())

    def test_body_is_the_channel_slice(self) -> None:
        expected = OscLightSender.float_to_uint8(
            OscLightSender._apply_levels(_frame().white, self.settings.curve, self.settings.lower_edge)
        )
        for i in range(FIRMWARE_NUM_CHUNKS):
            body = self.messages[i].dgram[OSC_PREAMBLE:]
            start = i * FIRMWARE_CHUNK_SIZE
            self.assertEqual(body, expected[start:start + FIRMWARE_CHUNK_SIZE].tobytes())


class ConfigMessageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = OscLightSenderSettings()
        self.settings.offsets.white_0 = 0
        self.settings.offsets.white_1 = 5
        self.settings.offsets.blue_0 = -10
        self.settings.offsets.blue_1 = 9
        self.messages = OscLightSender._build_config_messages(self.settings, 2000)

    def test_config_is_five_datagrams(self) -> None:
        self.assertEqual(
            [m.address for m in self.messages],
            ["/WS/o/0", "/WS/o/1", "/WS/o/2", "/WS/o/3", "/WS/r/0"],
        )

    def test_config_is_absent_from_the_pixel_burst(self) -> None:
        """Config is sent on change; keeping it out of the per-frame burst is the whole point."""
        chunks = OscLightSender._build_chunk_messages(
            _frame(), self.settings, FIRMWARE_CHUNK_SIZE, FIRMWARE_NUM_CHUNKS
        )
        assert chunks is not None
        self.assertFalse([m for m in chunks if m.address.startswith(("/WS/o/", "/WS/r/"))])

    def test_offset_value_lands_at_the_firmware_byte(self) -> None:
        """The firmware reads the int argument's low byte at `hdr[15]` as a signed char."""
        for message, expected in zip(self.messages, (0, 5, -10, 9)):
            self.assertEqual(len(message.dgram), 16)
            self.assertEqual(int.from_bytes(message.dgram[15:16], "big", signed=True), expected)

    def test_rpm_is_a_big_endian_int(self) -> None:
        """The firmware reassembles `/WS/r/0` from `hdr[12..15]`, MSB first."""
        rpm = self.messages[-1].dgram
        self.assertEqual(int.from_bytes(rpm[12:16], "big"), 2000)


if __name__ == "__main__":
    unittest.main()
