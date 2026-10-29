"""A camera frame as the contract hands it on: the image with its origin, kind and arrival time."""

from dataclasses import dataclass
from enum import Enum

import numpy as np


class FrameType(Enum):
    """The kinds of frame a source delivers. One so far: a backend with more (OAK4's depth,
    say) adds its own here, and the recorder's labels with them."""
    VIDEO = 1


@dataclass(frozen=True)
class CameraFrame:
    """One frame from one camera.

    `time` is the host's arrival time on the one clock, `time.perf_counter()`, stamped by the
    backend at hand-over; camera timestamps are not used, and everything downstream takes its
    time from the frame instead of reading a clock (the wall clock is for file names only).
    `image` is BGR uint8 (H, W, 3) or gray uint8 (H, W), as the backend delivers it.
    """
    cam_id: int
    kind: FrameType
    image: np.ndarray
    time: float
