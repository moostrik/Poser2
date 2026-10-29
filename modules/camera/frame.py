"""A camera frame as the contract hands it on: the image with its origin, kind and arrival time."""

from dataclasses import dataclass
from enum import Enum

import numpy as np


class FrameType(Enum):
    NONE_ = 0
    VIDEO = 1
    LEFT_ = 2
    RIGHT = 3
    DEPTH = 4


@dataclass(frozen=True)
class CameraFrame:
    """One frame from one camera.

    `time` is the host's arrival time, stamped by the backend at hand-over (`time.time()` until
    the time model of step 4.5 moves every stamp to `time.perf_counter()`); camera timestamps are
    not used. `image` is BGR uint8 (H, W, 3) or gray uint8 (H, W), as the backend delivers it.
    """
    cam_id: int
    kind: FrameType
    image: np.ndarray
    time: float
