"""Frame — the full per-tick render context and output.

Carries the clock snapshot (`tick`), motor state (`motor`), the LED pixel buffer (the POV
ring) and the four bar lights (the low regime). One object flows into every layer's `_draw`
and out to every consumer (board, the light/sound senders, render).

The bar lights are the discrete lamps on the rotating bar that the fixture drives directly
while it is in slot mode (commanded below ``FIXTURE_SLOW_RPM``, see ``motor.py``): the low
layers write them by name, the light sender maps them to the firmware's pixel slots, the
render simulates them as beams. This module describes them (id, output channel, beam
heading); the sender knows the slots, the render knows how to draw.
"""

import math
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Callable

import numpy as np

from .clock import Tick
from .motor import MotorState

BUFFER_DTYPE = np.float32


class BarLightId(IntEnum):
    """The four fixed lights on the bar, by strip position and colour."""
    FRONT_WHITE = 0
    BACK_WHITE  = auto()
    LEFT_BLUE   = auto()
    RIGHT_BLUE  = auto()


# Output channel per bar light (index = BarLightId): 0 = white, 1 = blue.
BAR_LIGHT_CHANNEL: np.ndarray = np.array([0, 0, 1, 1], dtype=np.int64)

# Beam heading of each bar light relative to the front white (radians, index = BarLightId),
# read off the firmware's fast-mode sampling offsets (firmware.cpp lines 277-280, of 3600):
# white 2 at +1800, blue 1 (the ``blue[0]`` slot, LEFT) at +2700, blue 2 (``blue[R//2]``,
# RIGHT) at +900. A strip index is an angle, so left trails the front by a quarter turn and
# right leads it by one. The ±10 px alignment corrections are not modelled.
BAR_LIGHT_HEADINGS: np.ndarray = np.array([0.0, math.pi, -math.pi / 2.0, math.pi / 2.0], dtype=np.float64)


@dataclass
class Frame:
    """Per-tick render context + light output. `white`/`blue` are views into `light_img`
    (the ring); `bar_lights` holds the four lamp levels, indexed by `BarLightId`."""
    resolution: int
    tick:       Tick
    motor:      MotorState              = field(default_factory=MotorState)
    playhead:   float                   = 0.0   # continuous content playhead (radians [-π,π), offset applied)
    light_img:  np.ndarray              = field(init=False)
    bar_lights: np.ndarray              = field(init=False)

    def __post_init__(self) -> None:
        # Shape (1, R, 3): channel 0 = white, channel 1 = blue, channel 2 = reserved
        self.light_img  = np.zeros((1, self.resolution, 3), dtype=BUFFER_DTYPE)
        self.bar_lights = np.zeros(len(BarLightId), dtype=BUFFER_DTYPE)

    @property
    def white(self) -> np.ndarray:
        """White channel."""
        return self.light_img[0, :, 0]

    @white.setter
    def white(self, value: np.ndarray | float) -> None:
        self.light_img[0, :, 0] = value

    @property
    def blue(self) -> np.ndarray:
        """Blue channel."""
        return self.light_img[0, :, 1]

    @blue.setter
    def blue(self, value: np.ndarray | float) -> None:
        self.light_img[0, :, 1] = value


FrameCallback = Callable[[Frame], None]
