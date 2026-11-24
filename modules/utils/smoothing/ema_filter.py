import math


class EmaFilter:
    """An exponential moving average of a single value, set by a time constant in seconds and corrected for the time
    between samples; an optional second time constant for falling values gives attack and release.

    `time_constant` and `fall_time_constant` may be set between calls; a time constant of 0 passes the value through.
    The first sample, after construction or `reset()`, is the starting value.
    """

    def __init__(self, time_constant: float, fall_time_constant: float | None = None) -> None:
        self.time_constant: float = time_constant               # s
        self.fall_time_constant: float | None = fall_time_constant     # s, for a value below the current one
        self._value: float = math.nan
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
            return self._value
        tau: float = self.time_constant
        if self.fall_time_constant is not None and value < self._value:
            tau = self.fall_time_constant
        if tau <= 0.0:
            self._value = value
        elif dt > 0.0:
            self._value += (value - self._value) * (1.0 - math.exp(-dt / tau))
        return self._value

    def reset(self, value: float = math.nan) -> None:
        """Start again from `value`; NaN starts at the next sample."""
        self._value = value
        self._timestamp = None
