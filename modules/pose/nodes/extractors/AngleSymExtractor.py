import math

import numpy as np

from ..Nodes import FilterNode
from ...features import Angles, AngleLandmark, AngleSymmetry, SymmetryElement
from ...frame import Frame, replace
from .LegDeviationExtractor import LegDeviationExtractorSettings


# The left and right joint of each joint element, in SymmetryElement order.
_LEFT: list[AngleLandmark] = [
    AngleLandmark.left_shoulder, AngleLandmark.left_elbow, AngleLandmark.left_hip, AngleLandmark.left_knee,
]
_RIGHT: list[AngleLandmark] = [
    AngleLandmark.right_shoulder, AngleLandmark.right_elbow, AngleLandmark.right_hip, AngleLandmark.right_knee,
]


class AngleSymExtractor(FilterNode):
    """Extracts the signed left-minus-right symmetry of every pair from the (mirrored) angles.

    The four joints: the difference wrapped to [-π, π), over π. ``arms``: the mean of the
    shoulder and elbow elements. ``legs``: the mean of the hip and knee differences, each over its
    full-deviation angle from the leg deviation's settings (``hip_degrees``, ``knee_degrees``), clipped to
    [-1, 1]. A pair's score is the lower of its two joints'; a whole-limb score the lower of its
    pairs'. Leaves the frame unchanged when no pair is valid.
    """

    def __init__(self, config: LegDeviationExtractorSettings | None = None) -> None:
        self._config = config if config is not None else LegDeviationExtractorSettings()

    def process(self, pose: Frame) -> Frame:
        angles = pose[Angles]
        diff = angles.values[_LEFT] - angles.values[_RIGHT]
        diff = np.arctan2(np.sin(diff), np.cos(diff))
        if not np.any(~np.isnan(diff)):
            return pose

        n = len(SymmetryElement)
        values = np.empty(n, dtype=np.float32)
        scores = np.empty(n, dtype=np.float32)
        pair_scores = np.minimum(angles.scores[_LEFT], angles.scores[_RIGHT])
        values[:4] = diff / math.pi
        scores[:4] = pair_scores
        values[SymmetryElement.arms] = (values[SymmetryElement.shoulder] + values[SymmetryElement.elbow]) / 2.0
        scores[SymmetryElement.arms] = min(pair_scores[SymmetryElement.shoulder], pair_scores[SymmetryElement.elbow])
        legs = (diff[SymmetryElement.hip] / math.radians(self._config.hip_degrees)
                + diff[SymmetryElement.knee] / math.radians(self._config.knee_degrees)) / 2.0
        values[SymmetryElement.legs] = min(max(legs, -1.0), 1.0) if not math.isnan(legs) else math.nan
        scores[SymmetryElement.legs] = min(pair_scores[SymmetryElement.hip], pair_scores[SymmetryElement.knee])
        scores[np.isnan(values)] = 0.0
        return replace(pose, {AngleSymmetry: AngleSymmetry(values, scores)})
