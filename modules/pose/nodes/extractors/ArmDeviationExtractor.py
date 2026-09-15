import math

import numpy as np

from ..Nodes import FilterNode
from ...features import Angles, AngleLandmark, ArmDeviation
from ...frame import Frame, replace
from modules.settings import BaseSettings, Field


class ArmDeviationExtractorSettings(BaseSettings):
    """The degree range the arm deviation spans: 0 up to ``min_degrees`` of arm movement, 1 from ``max_degrees``."""
    min_degrees: Field[float] = Field(18.0, min=0.0, max=180.0, step=1.0,
                                      description="Arm movement (°) up to which the deviation is 0: the arms hang neutral")
    max_degrees: Field[float] = Field(54.0, min=0.0, max=180.0, step=1.0,
                                      description="Arm movement (°) from which the deviation is 1: fully out of neutral")
    n_top:       Field[int]   = Field(1, min=1, max=4, step=1,
                                      description="Average the N most-moved arm joints (1 = the single most)")


# The arm joints the movement is taken over.
_ARM_JOINTS: list[AngleLandmark] = [
    AngleLandmark.left_shoulder, AngleLandmark.right_shoulder,
    AngleLandmark.left_elbow, AngleLandmark.right_elbow,
]


def range_position(degrees: float, low: float, high: float) -> float:
    """Where ``degrees`` sits in [low, high] as 0..1, clipped; a range of zero width is a step at ``low``."""
    if high <= low:
        return 1.0 if degrees > low else 0.0
    return min(max((degrees - low) / (high - low), 0.0), 1.0)


class ArmDeviationExtractor(FilterNode):
    """Extracts the arm deviation: how far the arms have moved out of neutral, as a position in a degree range.

    The angles are the calibrated ones (AngleCalibrator: 0 with the arm hanging, 0 with the elbow
    straight). The movement is the |angle| of the most-moved arm joint (the mean of the ``n_top`` most-moved),
    so one raised arm registers fully; the deviation is that movement's position between ``min_degrees``
    (0, standing neutral) and ``max_degrees`` (1, fully out of neutral), linear between. Leaves the frame
    unchanged when no arm angle is valid.
    """

    def __init__(self, config: ArmDeviationExtractorSettings | None = None) -> None:
        self._config = config if config is not None else ArmDeviationExtractorSettings()

    def process(self, pose: Frame) -> Frame:
        angles = pose[Angles]
        values = np.abs(angles.values[_ARM_JOINTS])
        valid = ~np.isnan(values)
        if not np.any(valid):
            return pose

        movement = values[valid]
        n = min(int(self._config.n_top), int(movement.size))
        top = np.partition(movement, -n)[-n:]
        deviation = range_position(math.degrees(float(np.mean(top))), self._config.min_degrees, self._config.max_degrees)
        score = float(np.mean(angles.scores[_ARM_JOINTS][valid]))
        return replace(pose, {ArmDeviation: ArmDeviation.from_value(deviation, score)})
