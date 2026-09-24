"""ArmTravelExtractor — the arm joints' travel from the calibrated angles, through dead zones.

The calibrated angles put neutral at 0 and the far pose at π (``AngleCalibrator``); a hanging arm
still reads a few degrees and a raised one stops short of π, so without dead zones the two fixed
points are never quite reached. The extractor takes each arm joint's magnitude from the band
between its neutral zone and its far zone onto 0..1, clamped, and keeps the sign: a value for the
instruments (the light and the sound), beside the angles, which stay raw for everything else.
"""

import math

import numpy as np

from ..Nodes import FilterNode
from ...features import Angles, AngleLandmark, ArmTravel
from ...frame import Frame, replace
from modules.settings import BaseSettings, Field


class ArmTravelExtractorSettings(BaseSettings):
    """The dead zones at both ends of each joint's travel, in degrees: a row per joint."""
    shoulder_neutral_dead_zone: Field[float] = Field(10.0, min=0.0, max=45.0, step=0.5, row_label="Shoulder", newline=True, label="Neutral Dead Zone", description="Shoulder angles within this of hanging read 0 (°)")
    shoulder_raised_dead_zone:  Field[float] = Field(10.0, min=0.0, max=45.0, step=0.5,                                      label="Raised Dead Zone",  description="Shoulder angles within this of raised read 1 (°)")
    elbow_neutral_dead_zone:    Field[float] = Field(10.0, min=0.0, max=45.0, step=0.5, row_label="Elbow",    newline=True, label="Neutral Dead Zone", description="Elbow angles within this of straight read 0 (°)")
    elbow_folded_dead_zone:     Field[float] = Field(10.0, min=0.0, max=45.0, step=0.5,                                      label="Folded Dead Zone",  description="Elbow angles within this of folded read 1 (°)")


# The arm joints, in TravelElement order.
_ARM_JOINTS: list[AngleLandmark] = [
    AngleLandmark.left_shoulder, AngleLandmark.right_shoulder,
    AngleLandmark.left_elbow, AngleLandmark.right_elbow,
]


class ArmTravelExtractor(FilterNode):
    """Extracts the signed travel of the four arm joints; see the module docstring.

    Per joint: ``|angle|`` from its neutral zone to π less its far zone, as 0..1, clamped; the sign
    of the angle is kept. A zero-width band is a step at the neutral zone. Leaves the frame
    unchanged when no arm angle is valid.
    """

    def __init__(self, config: ArmTravelExtractorSettings | None = None) -> None:
        self._config = config if config is not None else ArmTravelExtractorSettings()

    def process(self, pose: Frame) -> Frame:
        angles = pose[Angles]
        values = angles.values[_ARM_JOINTS].astype(np.float64)
        if not np.any(~np.isnan(values)):
            return pose

        C = self._config
        low = np.radians([C.shoulder_neutral_dead_zone, C.shoulder_neutral_dead_zone, C.elbow_neutral_dead_zone, C.elbow_neutral_dead_zone])
        high = math.pi - np.radians([C.shoulder_raised_dead_zone, C.shoulder_raised_dead_zone, C.elbow_folded_dead_zone, C.elbow_folded_dead_zone])
        magnitude = np.clip((np.abs(values) - low) / np.maximum(high - low, 1e-6), 0.0, 1.0)
        travel = (np.sign(values) * magnitude).astype(np.float32)          # NaN stays NaN
        scores = np.where(np.isnan(travel), 0.0, angles.scores[_ARM_JOINTS]).astype(np.float32)
        return replace(pose, {ArmTravel: ArmTravel(travel, scores)})
