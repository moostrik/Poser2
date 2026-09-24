"""ArmTravel — where each arm joint is along its travel from neutral to the far pose, with dead zones.

One value per arm joint, signed, in [-1, 1]: the magnitude is the joint's position along its travel,
0 inside the dead zone around neutral (the arm hanging, the elbow straight), 1 inside the dead zone
around the far pose (the shoulder raised, the elbow folded), linear between; the sign is the side of
the body the limb passes, as ``Angles`` has it, and means nothing to a consumer that takes the
absolute. The zones are the settings of ``ArmTravelExtractor``, which reads the calibrated angles
(0 neutral, π the far pose) and leaves ``Angles`` as they are, so the similarity, the symmetry and
the velocity keep the raw values and only the instruments read the travel: the light and the sound.
An element is NaN with score 0.0 when its angle is.
"""

from __future__ import annotations

from enum import IntEnum

from .base import BaseScalarFeature

TRAVEL_RANGE: tuple[float, float] = (-1.0, 1.0)


class TravelElement(IntEnum):
    """The four arm joints, in the order of their ``AngleLandmark`` entries."""
    left_shoulder  = 0
    right_shoulder = 1
    left_elbow     = 2
    right_elbow    = 3


class ArmTravel(BaseScalarFeature[TravelElement]):
    """Signed position of each arm joint along its travel, in [-1, 1]; see the module docstring."""

    @classmethod
    def enum(cls) -> type[IntEnum]:
        return TravelElement

    @classmethod
    def range(cls) -> tuple[float, float]:
        return TRAVEL_RANGE
