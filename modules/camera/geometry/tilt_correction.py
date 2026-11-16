"""The tilt correction: the homography that shows a tilted camera's frame as a level camera's, and boxes and
keypoints through it."""

import math

import numpy as np

from modules.settings import BaseSettings, Field, Widget
from modules.utils import Rect

TILT_STEP: float = 0.25             # ° the tilt moves before the frames are straightened anew
MIN_CORNER_SCALE: float = 0.1       # a frame corner past the horizon has no place in the level picture
# uv with its rows turned over, its own inverse: the frame's rows run opposite to the lens's and the IMU's.
FLIP_V: np.ndarray = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 1.0], [0.0, 0.0, 1.0]])


class TiltCorrectionSettings(BaseSettings):
    """Straightening the frames for the camera's tilt: the IMU's tilt plus `offset`."""
    enabled: Field[bool]  = Field(False, widget=Widget.switch, description="Straighten color and depth for the camera's tilt")
    offset:  Field[float] = Field(0.0, min=-60.0, max=60.0, step=0.5, widget=Widget.slider, description="Added to the IMU's tilt (°)")


def tilt_homography(intrinsics: list[list[float]], size: tuple[int, int], tilt: float) -> np.ndarray:
    """uv of a frame `size` to uv of the same frame from a level camera, for a camera `tilt` (°) up, its 3x3
    lens `intrinsics` in px, the level picture's height fitted to the frame's and centered across it; its third
    row is a pixel's depth scale."""
    width, height = size
    lens: np.ndarray = np.array(intrinsics, dtype=np.float64)
    lens /= lens[2, 2]
    cos_t, sin_t = math.cos(math.radians(tilt)), math.sin(math.radians(tilt))
    rotation: np.ndarray = np.array([[1.0, 0.0, 0.0], [0.0, cos_t, -sin_t], [0.0, sin_t, cos_t]])
    level: np.ndarray = lens @ rotation @ np.linalg.inv(lens)

    corners: np.ndarray = level @ np.array([[0.0, 0.0, 1.0], [width, 0.0, 1.0], [width, height, 1.0], [0.0, height, 1.0]]).T
    x, y = corners[:2] / np.maximum(corners[2], MIN_CORNER_SCALE)
    span_x, span_y = float(x.max() - x.min()), float(y.max() - y.min())
    scale: float = height / span_y
    fit: np.ndarray = np.array([[scale, 0.0, (width - scale * span_x) / 2.0 - scale * float(x.min())],
                                [0.0, scale, -scale * float(y.min())],
                                [0.0, 0.0, 1.0]])
    to_uv: np.ndarray = np.diag([1.0 / width, 1.0 / height, 1.0])
    return to_uv @ fit @ level @ np.diag([float(width), float(height), 1.0])


def homography_points(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    """uv points (N, 2) through a homography."""
    mapped: np.ndarray = homography @ np.column_stack((points, np.ones(len(points)))).T
    return (mapped[:2] / mapped[2]).T


def corrected_roi(roi: Rect, homography: np.ndarray) -> Rect:
    """A normalised box straightened with the frame: the box round its four moved corners, in the frame."""
    x0, y0, x1, y1 = roi.x, roi.y, roi.x + roi.width, roi.y + roi.height
    corners: np.ndarray = homography_points(homography, np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]))
    left, top = np.clip(corners.min(axis=0), 0.0, 1.0)
    right, bottom = np.clip(corners.max(axis=0), 0.0, 1.0)
    return Rect(float(left), float(top), float(right - left), float(bottom - top))


def corrected_keypoints(keypoints: np.ndarray | None, homography: np.ndarray) -> np.ndarray | None:
    """Keypoints (N, 3) straightened with the frame; the scores stay."""
    if keypoints is None:
        return None
    corrected: np.ndarray = keypoints.copy()
    corrected[:, :2] = homography_points(homography, keypoints[:, :2])
    return corrected


class TiltCorrection:
    """The current homography of a camera's frames; a new array only for a real change, so a consumer rebuilds
    what it derives only then."""

    def __init__(self, settings: TiltCorrectionSettings) -> None:
        self.settings: TiltCorrectionSettings = settings
        self.homography: np.ndarray | None = None
        self._tilt: float = math.nan        # the angle `homography` was made for

    def update(self, tilt: float, intrinsics: list[list[float]] | None, size: tuple[int, int]) -> None:
        """`tilt` is the IMU's, NaN without a reading; with no reading, no lens yet or the correction off, there
        is no homography and frames pass straight."""
        if not self.settings.enabled or intrinsics is None or math.isnan(tilt):
            self.homography = None
            self._tilt = math.nan
            return
        angle: float = tilt + self.settings.offset
        if math.isnan(self._tilt) or abs(angle - self._tilt) >= TILT_STEP:
            self._tilt = angle
            self.homography = FLIP_V @ tilt_homography(intrinsics, size, angle) @ FLIP_V
