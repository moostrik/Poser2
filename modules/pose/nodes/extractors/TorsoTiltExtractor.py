import math

import numpy as np

from ..Nodes import FilterNode
from ...features import Points2D, PointLandmark, TorsoTilt
from ...frame import Frame, replace
from modules.settings import BaseSettings, Field

# The spine's keypoints: shoulder pair (top) and hip pair (bottom).
_SHOULDERS: list[PointLandmark] = [PointLandmark.left_shoulder, PointLandmark.right_shoulder]
_HIPS:      list[PointLandmark] = [PointLandmark.left_hip,      PointLandmark.right_hip]
_SPINE:     list[PointLandmark] = _SHOULDERS + _HIPS

# A spine shorter than this (isotropic crop units) has no usable direction — the same
# ~2 % / ~4 px convention AngleUtils uses for too-close keypoints.
_MIN_SPINE: float = 0.02


class TorsoTiltExtractorSettings(BaseSettings):
    """Configuration for TorsoTiltExtractor."""
    tilt_rad:     Field[float] = Field(math.pi / 4.0, min=0.05, max=math.pi / 2.0, step=0.01,
                                       description="Spine angle from vertical (rad) that counts as full lean (±1.0)")
    aspect_ratio: Field[float] = Field(0.75, access=Field.INIT,
                                       description="Crop width/height — corrects the y axis to isotropic space (as the angle extractor)")


class TorsoTiltExtractor(FilterNode):
    """Extracts the signed sideways lean of the torso against the image vertical.

    Spine = hip midpoint → shoulder midpoint, with y scaled by 1/aspect_ratio exactly as
    ``AngleUtils.from_points`` does, so the angle is geometrically true. The signed angle from
    vertical (arctan2) is normalised by ``tilt_rad`` and clipped to [-1, 1]: 0 = upright,
    positive = shoulders toward image right of the hips. Forward/backward lean is not
    measured — in 2D it is only foreshortening. Leaves the frame unchanged when any spine
    keypoint is missing or the spine is degenerate.

    Assumes a level camera: the image vertical stands in for gravity.
    """

    def __init__(self, config: TorsoTiltExtractorSettings | None = None) -> None:
        self._config = config if config is not None else TorsoTiltExtractorSettings()

    def process(self, pose: Frame) -> Frame:
        points = pose[Points2D]
        if not points.are_valid(_SPINE):
            return pose

        ar_scale = np.array([1.0, 1.0 / self._config.aspect_ratio], dtype=np.float32)
        shoulders = np.mean(points.values[_SHOULDERS], axis=0) * ar_scale
        hips      = np.mean(points.values[_HIPS],      axis=0) * ar_scale
        spine = shoulders - hips                     # image y grows downward: upright → dy < 0
        if float(np.linalg.norm(spine)) < _MIN_SPINE:
            return pose

        angle = math.atan2(float(spine[0]), float(-spine[1]))   # 0 upright, +x = image right
        tilt = max(-1.0, min(1.0, angle / self._config.tilt_rad))
        score = float(min(points.get_scores(_SPINE)))
        return replace(pose, {TorsoTilt: TorsoTilt.from_value(tilt, score)})
