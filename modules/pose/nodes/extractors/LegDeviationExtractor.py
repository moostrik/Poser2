import math

import numpy as np

from ..Nodes import FilterNode
from ...features import Angles, AngleLandmark, LegDeviation
from ...frame import Frame, replace
from modules.settings import BaseSettings, Field


class LegDeviationExtractorSettings(BaseSettings):
    """The degree range each leg joint's deviation spans: 0 up to ``min_degrees`` of bend, 1 from its own top."""
    min_degrees:      Field[float] = Field(0.0,  min=0.0, max=180.0, step=1.0,
                                           description="Leg bend (°) up to which the deviation is 0: standing straight")
    hip_max_degrees:  Field[float] = Field(60.0, min=0.0, max=180.0, step=1.0,
                                           description="Hip bend (°) from which the hip counts as fully bent (1)")
    knee_max_degrees: Field[float] = Field(90.0, min=0.0, max=180.0, step=1.0,
                                           description="Knee bend (°) from which the knee counts as fully bent (1)")
    n_top:            Field[int]   = Field(2, min=1, max=4, step=1,
                                           description="Average the N most-bent leg joints (1 = the single most bent)")


# The leg joints, in the order the per-joint normalisation is applied.
_LEG_JOINTS: list[AngleLandmark] = [
    AngleLandmark.left_hip, AngleLandmark.right_hip,
    AngleLandmark.left_knee, AngleLandmark.right_knee,
]


class LegDeviationExtractor(FilterNode):
    """Extracts the joint-weighted leg deviation from the hip and knee angles.

    Each joint's |angle| becomes its position in a degree range: 0 up to the shared ``min_degrees`` of
    bend, 1 from its own top (``hip_max_degrees`` / ``knee_max_degrees`` — the knee flexes further for
    the same effort), linear between. The ``n_top`` most-bent joints are averaged (as
    AngleMotionExtractor does) so one bent leg registers fully rather than as a quarter of the pose.
    Leaves the frame unchanged when no leg angle is valid.
    """

    def __init__(self, config: LegDeviationExtractorSettings | None = None) -> None:
        self._config = config if config is not None else LegDeviationExtractorSettings()

    def process(self, pose: Frame) -> Frame:
        angles = pose[Angles]
        values = np.abs(angles.values[_LEG_JOINTS])
        valid = ~np.isnan(values)
        if not np.any(valid):
            return pose

        C = self._config
        tops = np.array([C.hip_max_degrees, C.hip_max_degrees, C.knee_max_degrees, C.knee_max_degrees], dtype=np.float32)
        bend = np.degrees(values[valid])
        span = np.maximum(tops[valid] - C.min_degrees, 1e-6)        # a zero-width range is a step at min
        deviation = np.clip((bend - C.min_degrees) / span, 0.0, 1.0)
        n = min(int(C.n_top), int(deviation.size))
        top = np.partition(deviation, -n)[-n:]
        score = float(np.mean(angles.scores[_LEG_JOINTS][valid]))
        return replace(pose, {LegDeviation: LegDeviation.from_value(float(np.mean(top)), score)})
