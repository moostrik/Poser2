"""A camera frame as the contract hands it on: the image with its origin, kind and arrival time."""

from dataclasses import dataclass
from enum import Enum

import numpy as np


class FrameType(Enum):
    """The kinds of frame a source delivers; a backend with more adds its own here, and the
    recorder's labels with them. Every frame callback gets every kind, so a consumer of one kind
    checks `kind`."""
    VIDEO = 1
    DEPTH = 2


@dataclass(frozen=True)
class CameraFrame:
    """One frame from one camera.

    `timestamp` is the host's arrival time on the one clock, `time.perf_counter()`, stamped by the
    backend at hand-over; camera timestamps are not used, and everything downstream takes its
    time from the frame instead of reading a clock (the wall clock is for file names only).
    `image` is BGR uint8 (H, W, 3) or gray uint8 (H, W), as the backend delivers it; with `nv12` it is
    packed NV12 uint8 (H * 3 / 2, W): the Y rows, then the interleaved UV rows, which a consumer
    converts where it is cheapest. A DEPTH frame's `image` is uint16 millimetres (H, W), 0 where
    there is no depth, on the same view as the VIDEO frames. `flip_h` says the image still has to
    be mirrored, because the device could not: the GPU upload does it, anything off the GPU takes
    `mirrored_image`. `homography` (`geometry.tilt_homography`, uv to uv) says the image still has
    to be straightened, before that mirror; one array stands for one correction, so a consumer
    keeps what it derives from it until the array changes.
    """
    cam_id: int
    kind: FrameType
    image: np.ndarray
    timestamp: float
    nv12: bool = False
    flip_h: bool = False
    homography: np.ndarray | None = None


def mirrored_image(frame: CameraFrame) -> np.ndarray:
    """The frame's image flipped left to right, in the frame's own layout. A new array: `Uploader`
    flips on the GPU instead, so this is for a consumer that keeps the image on the host."""
    image: np.ndarray = frame.image
    if not frame.nv12:
        return image[:, ::-1].copy()
    rows: int = image.shape[0] * 2 // 3
    width: int = image.shape[1]
    uv: np.ndarray = image[rows:].reshape(rows // 2, width // 2, 2)[:, ::-1]     # the pairs stay together
    return np.concatenate((image[:rows, ::-1], uv.reshape(rows // 2, width)))
