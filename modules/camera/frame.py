"""A camera frame as the contract hands it on: the image with its origin, kind and capture time."""

from dataclasses import dataclass, replace
from enum import Enum

import cv2
import numpy as np

from modules.utils import Point2f, Quad


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
    to_color: list[list[float]]             # the move from the depth camera to the video camera, 4x4, meters


@dataclass(frozen=True)
class CameraFrame:
    """One frame from one camera, as delivered: nothing straightened, rotated or flipped. Who shows it applies, in
    this order: the homography, `rotate90` (a transpose: rows become columns), `flip_v` (top to bottom), `flip_h`
    (left to right)."""
    cam_id: int
    kind: FrameType
    image: np.ndarray                       # BGR uint8 (H, W, 3) or gray (H, W); NV12 (H * 3 / 2, W) with `nv12`; DEPTH: uint16 mm, 0 none
    timestamp: float                        # its capture, s on `perf_counter`; its arrival where the backend cannot know it
    nv12: bool = False                      # packed NV12: the Y rows, then the interleaved UV rows
    homography: np.ndarray | None = None    # still to be straightened with, uv to uv, before `rotate90` and the flips
    rotate90: bool = False                  # still to be transposed: shown W x H as H x W
    flip_v: bool = False                    # still to be flipped top to bottom
    flip_h: bool = False                    # still to be flipped left to right
    readings: FrameReadings | None = None
    intrinsics: list[list[float]] | None = None     # its 3x3 lens, px
    alignment: DepthAlignment | None = None # a DEPTH frame on its own camera's view; None: on the video's already
    arrival: float | None = None            # its hand-over, s on `perf_counter`; None where `timestamp` is it
    confidence: np.ndarray | None = None    # DEPTH: uint8 (H, W) on the depth's view, higher surer; None where the source has none


def bgr_frame(frame: CameraFrame) -> CameraFrame:
    """The frame with its NV12 image converted to BGR, for a consumer that keeps the image on the host; any other
    frame as it is. The flags stay: `rotate90` would split NV12's interleaved chroma pairs, so it comes after this."""
    if not frame.nv12:
        return frame
    return replace(frame, image=cv2.cvtColor(frame.image, cv2.COLOR_YUV2BGR_NV12), nv12=False)


def oriented_image(frame: CameraFrame) -> np.ndarray:
    """The frame's image rotated and flipped as its flags say, the homography left out, in the frame's own layout:
    NV12 stays NV12, its chroma pairs kept pairs. A new array where a flag is set."""
    if not (frame.rotate90 or frame.flip_v or frame.flip_h):
        return frame.image
    if not frame.nv12:
        return np.ascontiguousarray(_oriented(frame.image, frame))
    rows: int = frame.image.shape[0] * 2 // 3
    width: int = frame.image.shape[1]
    y: np.ndarray = _oriented(frame.image[:rows], frame)
    uv: np.ndarray = _oriented(frame.image[rows:].reshape(rows // 2, width // 2, 2), frame)    # a pair is one pixel
    return np.concatenate((y, uv.reshape(uv.shape[0], uv.shape[1] * 2)))


def _oriented(image: np.ndarray, frame: CameraFrame) -> np.ndarray:
    """`image` (rows, columns, ...) rotated and flipped as the frame's flags say: a view."""
    if frame.rotate90:
        image = image.swapaxes(0, 1)
    if frame.flip_v:
        image = image[::-1]
    if frame.flip_h:
        image = image[:, ::-1]
    return image


def oriented_corners(frame: CameraFrame) -> Quad:
    """Where in the frame's image (0..1, the origin top-left) each corner of the oriented image comes from: its top
    left, top right, bottom right and bottom left, after `rotate90` and the flips."""
    def source(x: float, y: float) -> Point2f:
        if frame.flip_h:
            x = 1.0 - x
        if frame.flip_v:
            y = 1.0 - y
        return Point2f(y, x) if frame.rotate90 else Point2f(x, y)
    return Quad(source(0.0, 0.0), source(1.0, 0.0), source(1.0, 1.0), source(0.0, 1.0))
