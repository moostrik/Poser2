import math

import numpy as np

from ..Nodes import FilterNode
from ...features import Angles, AngleLandmark, LegDeviation
from ...frame import Frame, replace
from modules.settings import BaseSettings, Field


class LegDeviationExtractorSettings(BaseSettings):
    """Configuration for LegDeviationExtractor."""
    hip_degrees:  Field[float] = Field(60.0, min=5.0, max=180.0, step=1.0,
                                       description="Hip angle (°) that counts as fully deviated (1.0)")
    knee_degrees: Field[float] = Field(90.0, min=5.0, max=180.0, step=1.0,
                                       description="Knee angle (°) that counts as fully deviated (1.0)")
    n_top:        Field[int]   = Field(2, min=1, max=4, step=1,
                                       description="Average the N most-bent leg joints (1 = the single most bent)")


# The leg joints, in the order the per-joint normalisation is applied.
_LEG_JOINTS: list[AngleLandmark] = [
    AngleLandmark.left_hip, AngleLandmark.right_hip,
    AngleLandmark.left_knee, AngleLandmark.right_knee,
]


class LegDeviationExtractor(FilterNode):
    """Extracts the joint-weighted leg deviation from the hip and knee angles.

    Each joint's |angle| is normalised by its own full-deviation angle (``hip_degrees`` /
    ``knee_degrees`` — the knee flexes further for the same effort), clipped to [0, 1], and the
    ``n_top`` most-bent joints are averaged (as AngleMotionExtractor does) so one bent leg
    registers fully rather than as a quarter of the pose. Leaves the frame unchanged when
    no leg angle is valid.
    """

    def __init__(self, config: LegDeviationExtractorSettings | None = None) -> None:
        self._config = config if config is not None else LegDeviationExtractorSettings()

    def process(self, pose: Frame) -> Frame:
        angles = pose[Angles]
        values = np.abs(angles.values[_LEG_JOINTS])
        valid = ~np.isnan(values)
        if not np.any(valid):
            return pose

        norm = np.radians(np.array([self._config.hip_degrees, self._config.hip_degrees,
                                    self._config.knee_degrees, self._config.knee_degrees], dtype=np.float32))
        deviation = np.clip(values[valid] / norm[valid], 0.0, 1.0)
        n = min(int(self._config.n_top), int(deviation.size))
        top = np.partition(deviation, -n)[-n:]
        score = float(np.mean(angles.scores[_LEG_JOINTS][valid]))
        return replace(pose, {LegDeviation: LegDeviation.from_value(float(np.mean(top)), score)})
