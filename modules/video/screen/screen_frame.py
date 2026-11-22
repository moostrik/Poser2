"""ScreenFrame: one captured screen frame, handed from the screen capture to the screen recorder."""

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class ScreenFrame:
    """A finished frame in its `ScreenFormat`'s layout, (rows, width) of the format's `dtype`, in a buffer the capture
    lends: the Y plane, then Cb Cr, top row first. `timestamp` is its render tick's; `skipped` the frames the capture
    left out just before it, every buffer being lent. `release` gives the buffer back; the frame is not read after it."""
    data: np.ndarray
    timestamp: float
    skipped: int
    release: Callable[[], None]


ScreenFrameCallback = Callable[[ScreenFrame], None]
