"""The homography a source's frames carry: made from the IMU's tilt plus the offset, for a camera and
for a recording of one."""

import math

import numpy as np

from .geometry import tilt_homography
from ..settings import TiltCorrectionSettings

# How far (°) the tilt must move before the frames are straightened anew: the reading is noisy,
# and a picture that followed it would shimmer.
TILT_STEP: float = 0.25

# uv with its rows turned over, its own inverse: the frame's rows run opposite to the lens's and the
# IMU's, as seen on the rig.
FLIP_V: np.ndarray = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 1.0], [0.0, 0.0, 1.0]])


class TiltCorrection:
    """Keeps the current homography. `update` makes a new array only for a real change, and a
    consumer rebuilds what it derives only on a new array."""

    def __init__(self, settings: TiltCorrectionSettings) -> None:
        self.settings: TiltCorrectionSettings = settings
        self.homography: np.ndarray | None = None
        self._tilt: float = math.nan        # the angle `homography` was made for

    def update(self, tilt: float, intrinsics: list[list[float]] | None, size: tuple[int, int]) -> None:
        """`tilt` is the IMU's, NaN without a reading; with no reading, no lens yet or the
        correction off, there is no homography and frames pass straight."""
        if not self.settings.enabled or intrinsics is None or math.isnan(tilt):
            self.homography = None
            self._tilt = math.nan
            return
        angle: float = tilt + self.settings.offset
        if math.isnan(self._tilt) or abs(angle - self._tilt) >= TILT_STEP:
            self._tilt = angle
            self.homography = FLIP_V @ tilt_homography(intrinsics, size, angle) @ FLIP_V
