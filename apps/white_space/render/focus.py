"""The FOCUS layout's choice: which poses fill its slots. No GL, so it is unit-testable."""

import math

from modules.pose.features import Age
from modules.pose.frame import Frame

FOCUS_COLUMNS: int = 3


def focus_ids(frames: dict[int, Frame], count: int = FOCUS_COLUMNS) -> list[int]:
    """The ids of the ``count`` poses present longest, longest first, ties by id. A pose's age is
    its ``Age`` at LERP, the players' and the dummy's alike; a NaN age (a track's first frame after
    a reset) counts as 0."""
    def age(item: tuple[int, Frame]) -> float:
        value = item[1][Age].value
        return 0.0 if math.isnan(value) else value
    ordered = sorted(frames.items(), key=lambda item: (-age(item), item[0]))
    return [track_id for track_id, _ in ordered[:count]]
