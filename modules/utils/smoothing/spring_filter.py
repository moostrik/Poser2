import math


class SpringFilter:
    """A spring-damper that follows a single value, the one filter that overshoots.

    `responsiveness` is the natural frequency in Hz (higher is faster, typically 1-5); `damping` the damping ratio
    (0.7-1.0 is about critical, no overshoot; below is bouncy, above sluggish). The first sample, after construction or
    `reset()`, is the starting value.
    """

    def __init__(self, responsiveness: float = 2.0, damping: float = 0.8) -> None:
        if responsiveness <= 0:
            raise ValueError("responsiveness should be >0")
        if damping <= 0:
            raise ValueError("damping should be >0")
        self._responsiveness: float = float(responsiveness)
        self._damping: float = float(damping)
        self._target: float = math.nan
        self._value: float = math.nan
        self._velocity: float = 0.0
        self._timestamp: float | None = None

    @property
    def value(self) -> float:
        """The smoothed value; NaN before the first sample."""
        return self._value

    @property
    def target(self) -> float:
        """The last value it was given."""
        return self._target

    def __call__(self, value: float, timestamp: float) -> float:
        """The smoothed value after `value` at `timestamp` (s)."""
        dt: float = timestamp - self._timestamp if self._timestamp is not None else 0.0
        self._timestamp = timestamp
        self._target = float(value)
        if math.isnan(self._value):
            self._value = self._target
            self._velocity = 0.0
            return self._value
        if dt <= 0.0:
            return self._value
        omega: float = 2.0 * math.pi * self._responsiveness
        spring_force: float = omega * omega * (self._target - self._value)
        damping_force: float = 2.0 * self._damping * omega * self._velocity
        self._velocity += (spring_force - damping_force) * dt
        self._value += self._velocity * dt
        return self._value

    def set_responsiveness(self, responsiveness: float) -> None:
        """The natural frequency in Hz (> 0)."""
        if responsiveness <= 0:
            raise ValueError("responsiveness should be >0")
        self._responsiveness = float(responsiveness)

    def set_damping(self, damping: float) -> None:
        """The damping ratio (> 0)."""
        if damping <= 0:
            raise ValueError("damping should be >0")
        self._damping = float(damping)

    def set_parameters(self, responsiveness: float = 2.0, damping: float = 0.8) -> None:
        self.set_responsiveness(responsiveness)
        self.set_damping(damping)

    def reset(self, value: float = math.nan) -> None:
        """Start again from `value` at rest; NaN starts at the next sample."""
        self._value = value
        self._target = value
        self._velocity = 0.0
        self._timestamp = None
