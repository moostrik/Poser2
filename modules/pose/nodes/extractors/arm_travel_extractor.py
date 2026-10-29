"""ArmTravelExtractor — the arm joints' and the whole arms' travel from the calibrated angles,
through dead zones.

The calibrated angles put neutral at 0 and the far pose at π (``AngleCalibrator``); a hanging arm
still reads a few degrees and a raised one stops short of π, so without dead zones the two fixed
points are never quite reached. The extractor takes each arm joint's magnitude from the band
between its neutral zone and its far zone onto 0..1, clamped, and keeps the sign; from the
magnitudes it makes each whole arm, the weighted union ``s + (1 - s) × lift × e``
(``elbow_lift``), and their mean: values for the instruments (the light and the sound), beside
the angles, which stay raw for everything else.
"""

import math

import numpy as np

from ..nodes import FilterNode
from ...features import Angles, AngleLandmark, ArmTravel
from ...frame import Frame, replace
from modules.settings import BaseSettings, Field


class ArmTravelExtractorSettings(BaseSettings):
    """The dead zones at both ends of each joint's travel, in degrees, a row per joint; and the
    lift, the elbow's weight in the whole arm's union."""
    shoulder_neutral_dead_zone: Field[float] = Field(10.0, min=0.0, max=45.0, step=0.5, row_label="Shoulder", newline=True, label="Neutral Dead Zone", description="Shoulder angles within this of hanging read 0 (°)")
    shoulder_raised_dead_zone:  Field[float] = Field(10.0, min=0.0, max=45.0, step=0.5,                                      label="Raised Dead Zone",  description="Shoulder angles within this of raised read 1 (°)")
    elbow_neutral_dead_zone:    Field[float] = Field(10.0, min=0.0, max=45.0, step=0.5, row_label="Elbow",    newline=True, label="Neutral Dead Zone", description="Elbow angles within this of straight read 0 (°)")
    elbow_folded_dead_zone:     Field[float] = Field(10.0, min=0.0, max=45.0, step=0.5,                                      label="Folded Dead Zone",  description="Elbow angles within this of folded read 1 (°)")
    elbow_lift:                 Field[float] = Field(0.5,  min=0.0, max=1.0,  step=0.01, row_label="Arm",      newline=True, label="Elbow Lift",        description="How much a folded elbow lifts its arm (fraction of a full raise)")


# The arm joints, in TravelElement order.
_ARM_JOINTS: list[AngleLandmark] = [
    AngleLandmark.left_shoulder, AngleLandmark.right_shoulder,
    AngleLandmark.left_elbow, AngleLandmark.right_elbow,
]


class ArmTravelExtractor(FilterNode):
    """Extracts the signed travel of the four arm joints and the unsigned whole arms; see the
    module docstring.

    Per joint: ``|angle|`` from its neutral zone to π less its far zone, as 0..1, clamped; the sign
    of the angle is kept. A zero-width band is a step at the neutral zone. Per arm: the weighted
    union of its shoulder's and its elbow's magnitudes, and ``arms_mean`` the mean of the two; an arm
    with a missing joint is NaN with score 0.0, the others untouched. Leaves the frame unchanged
    when no arm angle is valid.
    """

    def __init__(self, settings: ArmTravelExtractorSettings | None = None) -> None:
        self.settings = settings if settings is not None else ArmTravelExtractorSettings()

    def process(self, pose: Frame) -> Frame:
        angles = pose[Angles]
        values = angles.values[_ARM_JOINTS].astype(np.float64)
        if not np.any(~np.isnan(values)):
            return pose

        C = self.settings
        low = np.radians([C.shoulder_neutral_dead_zone, C.shoulder_neutral_dead_zone, C.elbow_neutral_dead_zone, C.elbow_neutral_dead_zone])
        high = math.pi - np.radians([C.shoulder_raised_dead_zone, C.shoulder_raised_dead_zone, C.elbow_folded_dead_zone, C.elbow_folded_dead_zone])
        magnitude = np.clip((np.abs(values) - low) / np.maximum(high - low, 1e-6), 0.0, 1.0)
        joints = np.sign(values) * magnitude                                # NaN stays NaN
        s, e = magnitude[:2], magnitude[2:]                                 # the shoulders' and the elbows' magnitudes
        arm = s + (1.0 - s) * C.elbow_lift * e                              # NaN propagates into an arm and the mean
        travel = np.concatenate([joints, arm, [arm.mean()]]).astype(np.float32)
        joint_scores = angles.scores[_ARM_JOINTS]
        arm_scores = np.minimum(joint_scores[:2], joint_scores[2:])
        scores = np.concatenate([joint_scores, arm_scores, [arm_scores.min()]]).astype(np.float32)
        scores[np.isnan(travel)] = 0.0
        return replace(pose, {ArmTravel: ArmTravel(travel, scores)})
