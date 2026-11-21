"""hd_trio's LED bars: one Art-Net node drives a pair of vertical RGBW bars, both with the same bar fill, one on
each of two consecutive universes."""

from enum import IntEnum

import numpy as np

from modules.inout import ArtNetSender, ArtNetSenderSettings
from modules.settings import Field, Widget
from modules.utils import Color

CHANNELS_PER_PIXEL: int = 4     # R, G, B, W
BARS: int = 2                   # the pair, one universe each


class ChannelOrder(IntEnum):
    """Common RGBW channel orderings for different LED strips: every one that starts or ends with W."""
    # Ending with W
    RGBW = 0
    RBGW = 1
    GRBW = 2
    GBRW = 3
    BRGW = 4
    BGRW = 5
    # Starting with W
    WRGB = 6
    WRBG = 7
    WGRB = 8
    WGBR = 9
    WBRG = 10
    WBGR = 11


class ArtNetBarsSettings(ArtNetSenderSettings):
    channel_order: Field[ChannelOrder] = Field(ChannelOrder.RGBW, description="DMX channel order (e.g., RGBW, GRBW, WRGB)")
    pixels: Field[int]                 = Field(90, min=1, max=128, access=Field.INIT, widget=Widget.knob, description="Number of RGBW pixels per bar")
    color: Field[Color]                = Field(Color(1.0, 0.0, 0.0), widget=Widget.color, description="Accent color for lit pixels")
    white_strength: Field[float]       = Field(0.5, min=0.0, max=1.0, description="Base white intensity for all pixels")
    color_strength: Field[float]       = Field(0.5, min=0.0, max=1.0, description="Accent color intensity")
    bar: Field[float]                  = Field(0.5, min=0.0, max=1.0, description="Bar fill level (0=empty, 1=full)", access=Field.READWRITE)


class ArtNetBars:
    """Fills from the bottom (pixel 0) up to `bar`: lit pixels the accent color and the white, the rest the white.
    The sender sends without the flash, and black at start, when switched off and at stop."""

    def __init__(self, settings: ArtNetBarsSettings) -> None:
        self.settings: ArtNetBarsSettings = settings
        self.sender: ArtNetSender = ArtNetSender(settings, settings.pixels * CHANNELS_PER_PIXEL, self.fill,
                                                 universes=BARS, flash=False)

    def start(self) -> None:
        self.sender.start()

    def stop(self) -> None:
        self.sender.stop()

    def fill(self, packet: bytearray) -> None:
        """One bar's pixels, written to each bar's universe."""
        num_pixels: int = self.settings.pixels
        size: int = self.sender.packet_size
        c: Color = self.settings.color
        accent: tuple[int, int, int] = (int(c.r * 255), int(c.g * 255), int(c.b * 255))
        color_strength: float = self.settings.color_strength
        lit_count: int = min(max(int(self.settings.bar * num_pixels + 0.5), 0), num_pixels)    # round to nearest

        order: str = self.settings.channel_order.name
        pixels: np.ndarray = np.frombuffer(packet, dtype=np.uint8)[:num_pixels * CHANNELS_PER_PIXEL]
        pixels = pixels.reshape(num_pixels, CHANNELS_PER_PIXEL)
        for channel, value in zip('RGB', accent):
            pixels[:lit_count, order.index(channel)] = int(value * color_strength)
            pixels[lit_count:, order.index(channel)] = 0
        pixels[:, order.index('W')] = int(self.settings.white_strength * 255)
        for bar in range(1, BARS):
            packet[bar * size:(bar + 1) * size] = packet[:size]
