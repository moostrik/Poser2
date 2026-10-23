"""ShoulderDifferenceExtractor — which arm is the higher: the shoulders' signed travel difference.

Reads ``ArmTravel``, so the dead zones are already on the values, and takes the left shoulder's
magnitude less the right's: 0 with level shoulders, and exactly 0 at both fixed points through the
travel's zones, where ``AngleSymmetry``'s shoulder element, from the raw angles, never quite
rests (``ShoulderDifference``). Leaves the frame unchanged when either shoulder's travel is NaN.
"""

import math

from ..Nodes import FilterNode
from ...features import ArmTravel, TravelElement, ShoulderDifference
from ...frame import Frame, replace


class ShoulderDifferenceExtractor(FilterNode):
    """Extracts the shoulders' signed travel difference, left less right; see the module
    docstring."""

    def process(self, pose: Frame) -> Frame:
        travel = pose[ArmTravel]
        difference = (abs(float(travel.values[TravelElement.left_shoulder]))
                      - abs(float(travel.values[TravelElement.right_shoulder])))
        if math.isnan(difference):
            return pose
        score = min(float(travel.scores[TravelElement.left_shoulder]),
                    float(travel.scores[TravelElement.right_shoulder]))
        return replace(pose, {ShoulderDifference: ShoulderDifference.from_value(difference, score)})
