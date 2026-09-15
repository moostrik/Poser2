"""The POSE layout's slots: which poses fill them. No GL, so it is unit-testable."""

import math

from modules.pose.features import Age
from modules.pose.frame import Frame

POSE_SLOTS: int = 3


def longest_present(frames: dict[int, Frame], count: int = POSE_SLOTS) -> list[int]:
    """The ids of the ``count`` poses present longest, longest first, ties by id. A pose's age is
    its ``Age`` at LERP, the players' and the dummy's alike; a NaN age (a track's first frame after
    a reset) counts as 0."""
    def age(item: tuple[int, Frame]) -> float:
        value = item[1][Age].value
        return 0.0 if math.isnan(value) else value
    ordered = sorted(frames.items(), key=lambda item: (-age(item), item[0]))
    return [track_id for track_id, _ in ordered[:count]]


def pose_slots(slots: list[int | None], frames: dict[int, Frame]) -> list[int | None]:
    """The slots this frame, from last frame's: a pose keeps its slot while it is among the
    ``len(slots)`` longest present, so a neighbour leaving never moves it; a freed slot takes the
    longest-present pose not yet in one. ``None`` is an empty slot."""
    chosen = longest_present(frames, len(slots))
    kept: list[int | None] = [track_id if track_id in chosen else None for track_id in slots]
    newcomers = iter(track_id for track_id in chosen if track_id not in kept)
    return [track_id if track_id is not None else next(newcomers, None) for track_id in kept]
