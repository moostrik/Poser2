"""Frame — the full per-tick render context and output.

Carries the clock snapshot (`tick`), motor state (`motor`), and the LED pixel buffer.
One object flows into every layer's `_draw` and out to every consumer (board, osc_light,
osc_sound, render).
"""

import numpy as np
from typing import Callable
from dataclasses import dataclass, field

from .clock import Tick
from .motor import MotorState

BUFFER_DTYPE = np.float32


@dataclass
class Frame:
    """Per-tick render context + LED output. `white`/`blue` are views into `light_img`."""
    resolution: int
    tick:       Tick
    motor:      MotorState              = field(default_factory=MotorState)
    light_img:  np.ndarray              = field(init=False)

    def __post_init__(self) -> None:
        # Shape (1, R, 3): channel 0 = white, channel 1 = blue, channel 2 = reserved
        self.light_img = np.zeros((1, self.resolution, 3), dtype=BUFFER_DTYPE)

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
