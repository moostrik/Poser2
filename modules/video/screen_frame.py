"""ScreenFrame: one captured screen frame, handed from the screen capture to the screen recorder."""

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class ScreenFrame:
    """A finished NV12 frame, (height × 3/2, width) uint8, in a buffer the capture lends: the Y plane, then Cb Cr at
    half width and half height, top row first. `timestamp` is its render tick's; `skipped` the frames the capture left
    out just before it, every buffer being lent. `release` gives the buffer back; the frame is not read after it."""
    data: np.ndarray
    timestamp: float
    skipped: int
    release: Callable[[], None]

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    @property
    def height(self) -> int:
        return int(self.data.shape[0]) * 2 // 3


ScreenFrameCallback = Callable[[ScreenFrame], None]
