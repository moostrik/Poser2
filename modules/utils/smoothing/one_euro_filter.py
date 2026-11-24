import math


class OneEuroFilter:
    """The One Euro filter (Casiez et al. 2012) of a single value: a lowpass whose cutoff rises with the value's speed.

    `min_cutoff` (Hz) is the cutoff at rest, `beta` how much the cutoff rises per unit of speed, `d_cutoff` (Hz) the
    lowpass on the speed itself; each may be set between calls. The first sample, after construction or `reset()`, is
    the starting value.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0) -> None:
        self.min_cutoff: float = min_cutoff
        self.beta: float = beta
        self.d_cutoff: float = d_cutoff
        self._value: float = math.nan
        self._speed: float = 0.0            # the lowpassed speed, per second
        self._timestamp: float | None = None

    @property
    def value(self) -> float:
        """The smoothed value; NaN before the first sample."""
        return self._value

    def __call__(self, value: float, timestamp: float) -> float:
        """The smoothed value after `value` at `timestamp` (s)."""
        dt: float = timestamp - self._timestamp if self._timestamp is not None else 0.0
        self._timestamp = timestamp
        if math.isnan(self._value):
            self._value = value
            self._speed = 0.0
            return self._value
        if dt <= 0.0:
            return self._value
        speed: float = (value - self._value) / dt
        self._speed += (speed - self._speed) * _alpha(self.d_cutoff, dt)
        cutoff: float = self.min_cutoff + self.beta * abs(self._speed)
        self._value += (value - self._value) * _alpha(cutoff, dt)
        return self._value

    def reset(self, value: float = math.nan) -> None:
        """Start again from `value`; NaN starts at the next sample."""
        self._value = value
        self._speed = 0.0
        self._timestamp = None


def _alpha(cutoff: float, dt: float) -> float:
    """The smoothing factor of a first-order lowpass at `cutoff` (Hz) over `dt` (s)."""
    tau: float = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)
