"""The drag, both fluids': the share of the gap to a body's value the water closes in a step, and the speed below
which a body drags nothing."""

import math

DRAG_FLOOR: float = 0.02    # widths/s: the body's speed below which it drags nothing, so a dead trail pulls nothing toward 0


def drag_share(delta_time: float, drag_time: float) -> float:
    """The share of the gap to the body's velocity the water closes in one step of `delta_time` (s):
    1 − exp(−dt / drag_time), `drag_time` in ms; 0 ms sets the velocity at once."""
    if drag_time <= 0.0:
        return 1.0
    return 1.0 - math.exp(-delta_time * 1000.0 / drag_time)
