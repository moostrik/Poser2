"""ArmTravel — where each arm joint, and each whole arm, is along its travel from neutral to the
far pose, with dead zones.

The four joints are signed, in [-1, 1]: the magnitude is the joint's position along its travel,
0 inside the dead zone around neutral (the arm hanging, the elbow straight), 1 inside the dead zone
around the far pose (the shoulder raised, the elbow folded), linear between; the sign is the side of
the body the limb passes, as ``Angles`` has it, and means nothing to a consumer that takes the
absolute. The arms are unsigned, in [0, 1]: each arm is the weighted union of its joints'
magnitudes, ``s + (1 - s) × lift × e`` with the lift ``ArmTravelExtractor``'s ``elbow_lift`` — an
arm is lifted insofar as any of it is lifted, and with the lift under 1 only a raised shoulder
reaches 1 — and ``arms_mean`` is the mean of the two. The zones and the lift are the settings of
``ArmTravelExtractor``, which reads the calibrated angles (0 neutral, π the far pose) and leaves
``Angles`` as they are, so the similarity, the symmetry and the velocity keep the raw values and
only the instruments read the travel: the light and the sound. An element is NaN with score 0.0
when an angle it needs is.
"""

from __future__ import annotations

from enum import IntEnum

from .base import BaseScalarFeature

TRAVEL_RANGE: tuple[float, float] = (-1.0, 1.0)


class TravelElement(IntEnum):
    """The four arm joints, in the order of their ``AngleLandmark`` entries; then the whole arms."""
    left_shoulder  = 0
    right_shoulder = 1
    left_elbow     = 2
    right_elbow    = 3
    left_arm       = 4     # the union of its shoulder's and its elbow's magnitudes, unsigned
    right_arm      = 5
    arms_mean      = 6     # the mean of the two arms


class ArmTravel(BaseScalarFeature[TravelElement]):
    """Signed position of each arm joint along its travel, in [-1, 1], and the unsigned whole
    arms, in [0, 1]; see the module docstring."""

    @classmethod
    def enum(cls) -> type[IntEnum]:
        return TravelElement

    @classmethod
    def range(cls) -> tuple[float, float]:
        return TRAVEL_RANGE
