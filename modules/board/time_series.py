from __future__ import annotations

from enum import IntEnum
from threading import Lock
from typing import Protocol

import numpy as np

from modules.utils import SampleRing


class HasTimeSeries(Protocol):
    """Values over time, keyed by an app's IntEnum: any thread submits, the plot reads."""
    def submit_time_sample(self, series: IntEnum, value: float, timestamp: float) -> None: ...
    def get_time_series(self, window: float) -> tuple[float, dict[IntEnum, tuple[np.ndarray, np.ndarray]]]: ...


class TimeSeriesStoreMixin:
    """Thread-safe rings of (timestamp, value) samples, one per series."""

    def __init__(self) -> None:
        self._time_series_lock = Lock()
        self._time_series: dict[IntEnum, SampleRing] = {}
        self._time_series_end: float = 0.0

    def submit_time_sample(self, series: IntEnum, value: float, timestamp: float) -> None:
        with self._time_series_lock:
            ring: SampleRing | None = self._time_series.get(series)
            if ring is None:
                ring = self._time_series[series] = SampleRing()
            ring.add(timestamp, value)
            self._time_series_end = max(self._time_series_end, timestamp)

    def get_time_series(self, window: float) -> tuple[float, dict[IntEnum, tuple[np.ndarray, np.ndarray]]]:
        """The newest timestamp, and copies of each series' samples in the `window` seconds up to it."""
        with self._time_series_lock:
            end: float = self._time_series_end
            return end, {series: ring.since(end - window) for series, ring in self._time_series.items()}
