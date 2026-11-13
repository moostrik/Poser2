"""A camera frame as the contract hands it on: the image with its origin, kind and capture time."""

from dataclasses import dataclass
from enum import Enum

import numpy as np


CAPTURE_TOLERANCE: float = 0.002   # s: frames of one capture carry timestamps this close (OAK4: 0.12 ms apart)


class FrameType(Enum):
    """The kinds of frame a source delivers; every frame callback gets every kind."""
    VIDEO = 1
    DEPTH = 2


@dataclass(frozen=True)
class FrameReadings:
    """What the camera reported for one frame: tilt and roll (°, NaN without an IMU reading), exposure (µs), ISO."""
    tilt: float
    roll: float
    exposure: int
    iso: int


@dataclass(frozen=True)
class DepthAlignment:
    """Where a depth frame's pixels land in the video frame."""
    intrinsics: list[list[float]]           # the depth camera's 3x3 lens, px
    color_intrinsics: list[list[float]]     # the video camera's 3x3 lens, px
    color_size: tuple[int, int]             # the video frame's size
    to_color: list[list[float]]             # the move from the depth camera to the video camera, 4x4, metres


@dataclass(frozen=True)
class CameraFrame:
    """One frame from one camera, as delivered: nothing straightened, nothing mirrored."""
    cam_id: int
    kind: FrameType
    image: np.ndarray                       # BGR uint8 (H, W, 3) or gray (H, W); NV12 (H * 3 / 2, W) with `nv12`; DEPTH: uint16 mm, 0 none
    timestamp: float                        # its capture, s on `perf_counter`; its arrival where the backend cannot know it
    nv12: bool = False                      # packed NV12: the Y rows, then the interleaved UV rows
    flip_h: bool = False                    # still to be mirrored
    homography: np.ndarray | None = None    # still to be straightened with, uv to uv, before the mirror
    readings: FrameReadings | None = None
    intrinsics: list[list[float]] | None = None     # its 3x3 lens, px
    alignment: DepthAlignment | None = None # a DEPTH frame on its own camera's view; None: on the video's already
    arrival: float | None = None            # its hand-over, s on `perf_counter`; None where `timestamp` is it


def mirrored_image(frame: CameraFrame) -> np.ndarray:
    """The frame's image flipped left to right, in the frame's own layout. A new array: an uploader
    flips on the GPU instead, so this is for a consumer that keeps the image on the host."""
    image: np.ndarray = frame.image
    if not frame.nv12:
        return image[:, ::-1].copy()
    rows: int = image.shape[0] * 2 // 3
    width: int = image.shape[1]
    uv: np.ndarray = image[rows:].reshape(rows // 2, width // 2, 2)[:, ::-1]     # the pairs stay together
    return np.concatenate((image[:rows, ::-1], uv.reshape(rows // 2, width)))
