import math

import numpy as np

from ..Nodes import FilterNode
from ...features import Angles, AngleLandmark, ArmDeviation
from ...frame import Frame, replace
from modules.settings import BaseSettings, Field


class ArmDeviationExtractorSettings(BaseSettings):
    """Configuration for ArmDeviationExtractor."""
    shoulder_degrees: Field[float] = Field(90.0, min=5.0, max=180.0, step=1.0,
                                           description="Shoulder angle (°) that counts as fully deviated (1.0): the arm level")
    elbow_degrees:    Field[float] = Field(90.0, min=5.0, max=180.0, step=1.0,
                                           description="Elbow angle (°) that counts as fully deviated (1.0)")
    n_top:            Field[int]   = Field(1, min=1, max=4, step=1,
                                           description="Average the N most-deviated arm joints (1 = the single most)")


# The arm joints, in the order the per-joint normalisation is applied.
_ARM_JOINTS: list[AngleLandmark] = [
    AngleLandmark.left_shoulder, AngleLandmark.right_shoulder,
    AngleLandmark.left_elbow, AngleLandmark.right_elbow,
]


class ArmDeviationExtractor(FilterNode):
    """Extracts the joint-weighted arm deviation from the shoulder and elbow angles.

    The angles are the calibrated ones (AngleCalibrator: 0 with the arm hanging, 0 with the elbow
    straight). Each joint's |angle| is normalised by its own full-deviation angle (``shoulder_degrees`` /
    ``elbow_degrees``), clipped to [0, 1], and the ``n_top`` most-deviated joints are averaged (as
    LegDeviationExtractor does) so one raised arm registers fully rather than as a quarter of the
    pose. Leaves the frame unchanged when no arm angle is valid.
    """

    def __init__(self, config: ArmDeviationExtractorSettings | None = None) -> None:
        self._config = config if config is not None else ArmDeviationExtractorSettings()

    def process(self, pose: Frame) -> Frame:
        angles = pose[Angles]
        values = np.abs(angles.values[_ARM_JOINTS])
        valid = ~np.isnan(values)
        if not np.any(valid):
            return pose

        norm = np.radians(np.array([self._config.shoulder_degrees, self._config.shoulder_degrees,
                                    self._config.elbow_degrees, self._config.elbow_degrees], dtype=np.float32))
        deviation = np.clip(values[valid] / norm[valid], 0.0, 1.0)
        n = min(int(self._config.n_top), int(deviation.size))
        top = np.partition(deviation, -n)[-n:]
        score = float(np.mean(angles.scores[_ARM_JOINTS][valid]))
        return replace(pose, {ArmDeviation: ArmDeviation.from_value(float(np.mean(top)), score)})
