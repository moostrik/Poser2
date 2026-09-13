# Standard library imports
from dataclasses import dataclass
from enum import IntEnum, auto

# Local application imports
from ..tracker_base import TrackerAnnotation


class Rejection(IntEnum):
    """Why the tracker did not count a detection this frame — one member per filter in the intake.

    Carried on the annotation so the panorama can draw the rejected detection and name the filter,
    rather than it vanishing.
    """
    YOUNG = auto()       # the device has not held it for `age_filter` frames yet
    SMALL = auto()       # its box is shorter than `height_filter`
    DEAD_ZONE = auto()   # a new arrival inside `seam.dead_zone` of a field edge
    PAST_EDGE = auto()   # past `rig.zone_max_radius`, with `zone_filter` on
    NO_ID = auto()       # a new person while every world id is in use


@dataclass(frozen=True)
class Annotation(TrackerAnnotation):
    """What one camera's box says about one person, all of it derived in `Rig`.

    `distance` is from that camera (m); `height` is absolute (m) and needs no re-projection,
    since the same camera sees the person's feet and head. `rejected` is set when a filter
    rejected this detection (or, for a person already tracked, stopped counting it).
    """
    local_angle: float
    world_angle: float
    overlap: bool
    distance: float = 0.0
    height: float = 0.0
    rejected: Rejection | None = None
