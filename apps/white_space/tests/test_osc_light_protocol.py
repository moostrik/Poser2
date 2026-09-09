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
from apps.white_space.light import Frame, Tick, BarLightId

# The firmware reads a fixed-size OSC preamble before the pixel body: 12-byte padded address,
# 4-byte typetag, 4-byte blob length.
OSC_PREAMBLE = 20
RESOLUTION = 3600


def _frame() -> Frame:
    frame = Frame(RESOLUTION, Tick(0.0, 0.0))
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
            _frame(), self.settings, FIRMWARE_CHUNK_SIZE, FIRMWARE_NUM_CHUNKS, slow=False
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
            OscLightSender._apply_levels(_frame().white, self.settings.curve,
                                         self.settings.lower_edge, self.settings.upper_edge)
        )
        for i in range(FIRMWARE_NUM_CHUNKS):
            body = self.messages[i].dgram[OSC_PREAMBLE:]
            start = i * FIRMWARE_CHUNK_SIZE
            self.assertEqual(body, expected[start:start + FIRMWARE_CHUNK_SIZE].tobytes())


class LevelWindowTest(unittest.TestCase):
    """The output mapping targets the analog LED driver's usable window: dark below the
    forward threshold (`lower_edge`), saturated above the ceiling (`upper_edge`)."""

    def test_lit_pixels_map_into_the_usable_window(self) -> None:
        x = np.array([0.0, 0.001, 0.5, 1.0], dtype=np.float32)
        out = OscLightSender._apply_levels(x, curve=1.0, lower_edge=0.32, upper_edge=0.78)
        self.assertEqual(out[0], 0.0)                       # true black stays off
        self.assertGreater(out[1], 0.32)                    # lit → lifted past the floor
        self.assertAlmostEqual(float(out[2]), 0.32 + 0.46 * 0.5, places=6)
        self.assertAlmostEqual(float(out[3]), 0.78, places=6)   # full maps to the ceiling
        self.assertTrue(np.all(np.diff(out) >= 0.0))        # monotonic

    def test_crossed_edges_collapse_to_the_floor(self) -> None:
        out = OscLightSender._apply_levels(np.array([1.0], dtype=np.float32),
                                           curve=1.0, lower_edge=0.4, upper_edge=0.2)
        self.assertAlmostEqual(float(out[0]), 0.4, places=6)   # never inverts


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
            _frame(), self.settings, FIRMWARE_CHUNK_SIZE, FIRMWARE_NUM_CHUNKS, slow=False
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


class FixtureSlotTest(unittest.TestCase):
    """The fixture's readout mode follows the rpm sent with the frame: in slow mode it reads the
    four bar lights from pixel 0 and the middle pixel of each channel and nothing else, in ring
    mode it steps the ring and never reads a slot. The rebuild mirrors that exactly."""

    LEVELS = {BarLightId.FRONT_WHITE: 0.9, BarLightId.BACK_WHITE: 0.6,
              BarLightId.LEFT_BLUE: 0.4, BarLightId.RIGHT_BLUE: 0.2}
    HALF = RESOLUTION // 2

    def _lit_frame(self) -> Frame:
        frame = _frame()
        for light, level in self.LEVELS.items():
            frame.bar_lights[light] = level
        return frame

    @staticmethod
    def _bodies(frame: Frame, slow: bool) -> list[bytes]:
        messages = OscLightSender._build_chunk_messages(
            frame, OscLightSenderSettings(), FIRMWARE_CHUNK_SIZE, FIRMWARE_NUM_CHUNKS, slow
        )
        assert messages is not None
        return [m.dgram[OSC_PREAMBLE:] for m in messages]

    def test_slow_mode_puts_the_bar_lights_in_the_slots(self) -> None:
        white, blue = OscLightSender._rebuild_fixture_pixels(self._lit_frame(), slow=True)
        self.assertAlmostEqual(float(white[0]),         0.9, places=6)   # front white
        self.assertAlmostEqual(float(white[self.HALF]), 0.6, places=6)   # back white
        self.assertAlmostEqual(float(blue[0]),          0.4, places=6)   # left blue
        self.assertAlmostEqual(float(blue[self.HALF]),  0.2, places=6)   # right blue
        np.testing.assert_array_equal(white[1:self.HALF], _frame().white[1:self.HALF])   # the rest is the ring

    def test_slow_mode_replaces_ring_content_at_the_slots(self) -> None:
        frame = self._lit_frame()
        frame.white[0], frame.blue[self.HALF] = 1.0, 1.0        # ring content the fixture never reads
        white, blue = OscLightSender._rebuild_fixture_pixels(frame, slow=True)
        self.assertAlmostEqual(float(white[0]), 0.9, places=6)
        self.assertAlmostEqual(float(blue[self.HALF]), 0.2, places=6)

    def test_ring_mode_sends_the_ring_and_drops_the_bar_lights(self) -> None:
        frame = self._lit_frame()
        white, blue = OscLightSender._rebuild_fixture_pixels(frame, slow=False)
        np.testing.assert_array_equal(white, frame.white)
        np.testing.assert_array_equal(blue, frame.blue)

    def test_rebuild_never_mutates_the_shared_frame(self) -> None:
        frame = self._lit_frame()
        before = frame.light_img.copy()
        OscLightSender._rebuild_fixture_pixels(frame, slow=True)
        np.testing.assert_array_equal(frame.light_img, before)

    def test_wire_is_identical_to_the_baked_in_pixel_model(self) -> None:
        """Regression against the previous data model, where the low layers wrote the lamp
        levels straight into the slot pixels: the same look must produce the same bytes."""
        baked = _frame()
        baked.white[0], baked.white[self.HALF] = 0.9, 0.6
        baked.blue[0],  baked.blue[self.HALF]  = 0.4, 0.2
        explicit = self._lit_frame()
        self.assertEqual(self._bodies(explicit, slow=True), self._bodies(baked, slow=False))

    def test_blackout_is_dark_in_slow_mode_too(self) -> None:
        zero = Frame(RESOLUTION, Tick(0.0, 0.0))
        for body in self._bodies(zero, slow=True):
            self.assertEqual(body, bytes(FIRMWARE_CHUNK_SIZE))


class BlackoutTest(unittest.TestCase):
    """The shutdown blackout: rpm 0 → six all-zero chunks (blue2 last) → rpm 0 again,
    so a clean quit leaves the fixture dark and stopped without waiting for the
    firmware's Ethernet watchdog."""

    def setUp(self) -> None:
        self.messages = OscLightSender._build_blackout_messages(
            OscLightSenderSettings(), FIRMWARE_CHUNK_SIZE, FIRMWARE_NUM_CHUNKS
        )

    def test_rpm_zero_brackets_the_dark_frame(self) -> None:
        self.assertEqual(
            [m.address for m in self.messages],
            ["/WS/r/0", "/WS/white0", "/WS/white1", "/WS/white2",
             "/WS/blue0", "/WS/blue1", "/WS/blue2", "/WS/r/0"],
        )
        for rpm in (self.messages[0], self.messages[-1]):
            self.assertEqual(int.from_bytes(rpm.dgram[12:16], "big"), 0)

    def test_pixel_bodies_are_all_zero(self) -> None:
        for message in self.messages[1:-1]:
            body = message.dgram[OSC_PREAMBLE:]
            self.assertEqual(len(body), FIRMWARE_CHUNK_SIZE, message.address)
            self.assertEqual(body, bytes(FIRMWARE_CHUNK_SIZE), message.address)


if __name__ == "__main__":
    unittest.main()
