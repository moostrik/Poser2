"""ShoulderDifference — how unequal the two shoulders' travels are, signed, in [-1, 1].

The left shoulder's travel magnitude less the right's (``ArmTravel``, the absolutes taken), so
which arm is the higher: 0 with level shoulders, +1 the left straight up with the right hanging,
−1 the reverse. It is not ``AngleSymmetry``'s shoulder element, which compares the raw angles
over π with no dead zones and so never quite rests at 0; the travels' dead zones make the
difference rest exactly there at both fixed points. Populated by ``ShoulderDifferenceExtractor``.
Absent (NaN, score 0.0) when either shoulder's travel is.
"""

from __future__ import annotations

from .base import SingleValue

SHOULDER_DIFFERENCE_RANGE: tuple[float, float] = (-1.0, 1.0)


class ShoulderDifference(SingleValue):
    """The shoulders' signed travel difference, left less right, in [-1, 1]; see the module
    docstring."""

    @classmethod
    def range(cls) -> tuple[float, float]:
        return SHOULDER_DIFFERENCE_RANGE
