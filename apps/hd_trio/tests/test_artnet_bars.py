"""Tests for the Art-Net bars' packet: one bar's pixels, lit from the bottom, the same on both bars' universes (the
bars are never started: no network)."""

import unittest

from apps.hd_trio.artnet_bars import ArtNetBars, ArtNetBarsSettings, ChannelOrder
from modules.utils import Color


def bars(pixels: int, order: ChannelOrder, color: Color, white: float, strength: float, bar: float) -> ArtNetBars:
    settings = ArtNetBarsSettings()
    settings.pixels = pixels
    settings.channel_order = order
    settings.color = color
    settings.white_strength = white
    settings.color_strength = strength
    settings.bar = bar
    return ArtNetBars(settings)


def expected(pixels: int, order: ChannelOrder, color: Color, white: float, strength: float, bar: float) -> bytes:
    """The buffer as the pixel loop writes it: lit pixels the accent color and the white, the rest the white."""
    lit = int(bar * pixels + 0.5)
    w = int(white * 255)
    accent = (int(int(color.r * 255) * strength), int(int(color.g * 255) * strength), int(int(color.b * 255) * strength))
    out = bytearray(pixels * 4)
    for i in range(pixels):
        rgbw = (*accent, w) if i < lit else (0, 0, 0, w)
        for channel, value in zip('RGBW', rgbw):
            out[i * 4 + order.name.index(channel)] = value
    return bytes(out)


CASES = [
    # pixels, channel order, color, white, color strength, bar
    (90, ChannelOrder.RGBW, Color(1.0, 0.0, 0.0), 0.5, 0.5, 0.5),
    (3,  ChannelOrder.GRBW, Color(0.2, 0.6, 1.0), 0.0, 1.0, 0.5),       # 1.5 pixels lit: rounds to 2
    (8,  ChannelOrder.WRGB, Color(1.0, 1.0, 1.0), 1.0, 0.25, 0.0),      # empty: the white alone
    (8,  ChannelOrder.BGRW, Color(0.5, 0.3, 0.9), 0.3, 0.7, 1.0),       # full
    (7,  ChannelOrder.WBGR, Color(0.1, 0.9, 0.4), 0.8, 0.9, 0.5),       # 3.5 pixels lit: rounds to 4
    (128, ChannelOrder.GBRW, Color(0.33, 0.66, 0.99), 0.12, 0.61, 0.37),
    (1,  ChannelOrder.WGBR, Color(0.7, 0.2, 0.5), 0.6, 0.4, 0.49),      # half a pixel short: unlit
    (1,  ChannelOrder.RBGW, Color(0.7, 0.2, 0.5), 0.6, 0.4, 0.5),       # half a pixel: lit
]


def filled(b: ArtNetBars) -> bytes:
    b.fill(b.sender.packet)
    return bytes(b.sender.packet)


class TestArtNetBarsFrame(unittest.TestCase):
    def test_each_bar_is_lit_from_the_bottom(self):
        for case in CASES:
            with self.subTest(case=case):
                self.assertEqual(filled(bars(*case)), expected(*case) * 2)

    def test_the_loop_s_known_bytes(self):
        # two pixels lit of three, GRBW: green, red, blue, white
        b = bars(3, ChannelOrder.GRBW, Color(0.2, 0.6, 1.0), 0.0, 1.0, 0.5)
        self.assertEqual(filled(b), bytes([153, 51, 255, 0, 153, 51, 255, 0, 0, 0, 0, 0]) * 2)

    def test_the_bars_go_on_two_universes_without_the_flash(self):
        b = bars(3, ChannelOrder.GRBW, Color(0.2, 0.6, 1.0), 0.0, 1.0, 0.5)
        self.assertEqual(b.sender.universes, 2)
        self.assertFalse(b.sender.flash)


if __name__ == '__main__':
    unittest.main()
