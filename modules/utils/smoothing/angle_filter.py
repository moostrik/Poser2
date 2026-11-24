import math
from typing import Protocol


class Filter(Protocol):
    """A single-value filter as `AngleFilter` uses it."""
    @property
    def value(self) -> float: ...
    def __call__(self, value: float, timestamp: float) -> float: ...
    def reset(self, value: float = math.nan) -> None: ...


class AngleFilter:
    """A filter of an angle in radians: each angle moves to the turn nearest the filter's value (the shortest path)
    before the filter runs, and its value is wrapped to ±π. A rise is a positive turn and a fall a negative one."""

    def __init__(self, filter: Filter) -> None:
        self._filter: Filter = filter

    @property
    def value(self) -> float:
        """The smoothed angle in ±π; NaN before the first sample."""
        return _wrap(self._filter.value)

    def __call__(self, angle: float, timestamp: float) -> float:
        """The smoothed angle in ±π after `angle` (radians) at `timestamp` (s)."""
        current: float = self._filter.value
        target: float = _wrap(angle) if math.isnan(current) else current + _wrap(angle - current)
        return _wrap(self._filter(target, timestamp))

    def reset(self, angle: float = math.nan) -> None:
        """Start again from `angle`; NaN starts at the next sample."""
        self._filter.reset(_wrap(angle))


def _wrap(angle: float) -> float:
    """The angle in [-π, π)."""
    return (angle + math.pi) % math.tau - math.pi
