"""GhostState — a White Space-native pose feature marking a ghost frame as PASSIVE or ACTIVE.

A ghost is an ordinary pose ``Frame``; its lifecycle state rides on the frame as this feature (like
``Fade`` / ``PlayheadOffset``). A frame **without** ``GhostState`` is a live pose. Consumers filter by it:
only ``ACTIVE`` ghosts are sent to OSC; the light reads the state to render active (and, later, passive)
ghosts. The Frame ECS stores scalar features, so the enum rides as the single value.
"""

from __future__ import annotations

from enum import IntEnum

from modules.pose.features import SingleValue
from modules.pose.frame import Frame


class GhostStateValue(IntEnum):
    """Lifecycle of a ghost."""
    PASSIVE = 0   # follows its live person, building dwell/motion; not sent to OSC
    ACTIVE  = 1   # broke free from a settled spot; frozen, faded, reclaimable, sent to OSC


class GhostState(SingleValue):
    """A ghost's ``GhostStateValue`` carried as a single scalar. Absent (NaN) ⇒ a live pose."""

    @classmethod
    def range(cls) -> tuple[float, float]:
        return (0.0, float(max(GhostStateValue)))


def ghost_state(frame: Frame) -> GhostStateValue | None:
    """The frame's ``GhostState``, or ``None`` when it carries none (a live pose)."""
    if GhostState in frame:
        return GhostStateValue(int(frame[GhostState].value))
    return None
