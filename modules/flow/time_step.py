"""The fluid's time step: the measured render rate smoothed by an Ema, or a fixed rate, at most
MAX_STEP; and the viscosity solve's iterations for a step."""

import math
from enum import IntEnum


class TimeStepMode(IntEnum):
    MEASURED = 0    # follows the render rate, smoothed over `fps_smoothing` seconds
    FIXED =    1    # one step of 1 / `fixed_fps` per frame, whatever the render rate: runs compare


MIN_TICK: float = 1.0 / 240.0   # s: a tick measured shorter is a clock step, not a frame
MAX_STEP: float = 1.0 / 20.0    # s: the longest step the fluid takes; a slower frame runs it slower than real time
REFERENCE_STEP: float = 1.0 / 60.0  # s: the step a solver's iteration setting is for
TICK_BAND: float = 2.0          # a tick is held to this share of the mean tick: a stall counts as one long frame
TICK_WEIGHT: float = 0.05       # weight of a tick on the mean tick's Ema


def hold_tick(dt: float, mean: float | None) -> tuple[float, float]:
    """A render tick's measured length `dt` (s) as the flow steps it, and the mean tick one tick on:
    `dt` held to at least MIN_TICK and at most TICK_BAND times `mean`, an Ema of the ticks, so a
    stall counts as one long frame and the rest is dropped; the first tick (`mean` None) sets the
    mean. A `dt` that is not a tick's (not finite) takes the mean."""
    if mean is not None and not (math.isfinite(mean) and mean > 0.0):
        mean = None
    if not math.isfinite(dt):
        dt = mean if mean is not None else MIN_TICK
    dt = max(dt, MIN_TICK)
    if mean is None:
        return dt, dt
    dt = min(dt, TICK_BAND * mean)
    return dt, mean + TICK_WEIGHT * (dt - mean)


def viscosity_iterations(iterations: int, dt: float) -> int:
    """The viscosity solve's iterations for a step of `dt`: the setting's at a 60 fps step, fewer for a
    shorter one, which diffuses less, never more for a longer one, since a slow frame given more work
    renders slower still."""
    return max(1, round(iterations * min(dt / REFERENCE_STEP, 1.0)))


class TimeStep:
    """dt per tick. Plain arithmetic, no clock: the caller hands in the tick's dt it measured."""

    def __init__(self, fps: float = 60.0) -> None:
        self._dt: float = 1.0 / fps

    @property
    def dt(self) -> float:
        return self._dt

    def update(self, mode: TimeStepMode, dt: float, fixed_fps: float, smoothing: float) -> float:
        """Advance one tick and return its dt: `dt`, the tick's measured length (s), through the Ema,
        or a fixed step, at most MAX_STEP. `smoothing` is the Ema's time constant in seconds; 0
        follows each tick."""
        if mode == TimeStepMode.FIXED:
            self._dt = min(1.0 / max(1.0, fixed_fps), MAX_STEP)
            return self._dt
        raw: float = min(max(dt, 0.0), MAX_STEP)
        if smoothing <= 0.0:
            self._dt = raw
        else:
            k: float = 1.0 - math.exp(-raw / smoothing)     # weighted by the frame's own length
            self._dt += (raw - self._dt) * k
        return self._dt
