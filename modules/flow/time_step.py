"""The fluid's time step: the measured render rate smoothed by an Ema, or a fixed rate."""

import math
from enum import IntEnum


class TimeStepMode(IntEnum):
    MEASURED = 0    # follows the render rate, smoothed over `fps_smoothing` seconds
    FIXED =    1    # one step of 1 / `fixed_fps` per frame, whatever the render rate: runs compare


class TimeStep:
    """dt per tick. Plain arithmetic, no clock: the caller hands in the rate it measured."""

    def __init__(self, fps: float = 60.0) -> None:
        self._dt: float = 1.0 / fps

    @property
    def dt(self) -> float:
        return self._dt

    def update(self, mode: TimeStepMode, measured_fps: float, fixed_fps: float, smoothing: float) -> float:
        """Advance one tick and return its dt. `smoothing` is the Ema's time constant in seconds; 0 follows each tick."""
        if mode == TimeStepMode.FIXED:
            self._dt = 1.0 / max(1.0, fixed_fps)
            return self._dt
        raw: float = 1.0 / max(1.0, measured_fps)
        if smoothing <= 0.0:
            self._dt = raw
        else:
            k: float = 1.0 - math.exp(-raw / smoothing)     # weighted by the frame's own length
            self._dt += (raw - self._dt) * k
        return self._dt
