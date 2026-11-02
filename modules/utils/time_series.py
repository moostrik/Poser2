"""Values over time: a ring of (timestamp, value) samples, and its binning into the columns of a plot."""

import numpy as np


class SampleRing:
    """The last `capacity` samples of one series. Not locked: the store that holds it locks."""

    def __init__(self, capacity: int = 4096) -> None:
        self._timestamps: np.ndarray = np.zeros(capacity, np.float64)
        self._values: np.ndarray = np.zeros(capacity, np.float64)
        self._next: int = 0
        self._count: int = 0

    def __len__(self) -> int:
        return self._count

    def add(self, timestamp: float, value: float) -> None:
        self._timestamps[self._next] = timestamp
        self._values[self._next] = value
        self._next = (self._next + 1) % len(self._timestamps)
        self._count = min(self._count + 1, len(self._timestamps))

    def since(self, timestamp: float) -> tuple[np.ndarray, np.ndarray]:
        """Copies of the samples at or after `timestamp`, oldest first."""
        capacity: int = len(self._timestamps)
        order: np.ndarray = (np.arange(self._count) + self._next - self._count) % capacity
        timestamps: np.ndarray = self._timestamps[order]
        keep: np.ndarray = timestamps >= timestamp
        return timestamps[keep], self._values[order][keep]


def bin_samples(timestamps: np.ndarray, values: np.ndarray, end: float, window: float, columns: int) -> tuple[np.ndarray, np.ndarray]:
    """The lowest and highest value in each of `columns` equal slices of the `window` seconds up to `end`.

    A column without samples is NaN in both, so a plot can leave it out; the highest keeps a single
    spike visible however many samples share its column.
    """
    low: np.ndarray = np.full(columns, np.inf)
    high: np.ndarray = np.full(columns, -np.inf)
    column: np.ndarray = np.floor((timestamps - (end - window)) / window * columns).astype(np.int64)
    inside: np.ndarray = (column >= 0) & (column < columns)
    np.minimum.at(low, column[inside], values[inside])
    np.maximum.at(high, column[inside], values[inside])
    empty: np.ndarray = ~np.isfinite(low)
    low[empty] = np.nan
    high[empty] = np.nan
    return low, high


def hold_gaps(low: np.ndarray, high: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Empty columns take the last sampled one's value: a series slower than the columns reads as a line, not a comb.

    Columns before the first sample stay empty.
    """
    sampled: np.ndarray = np.isfinite(low)
    last: np.ndarray = np.maximum.accumulate(np.where(sampled, np.arange(len(low)), -1))
    held: np.ndarray = last >= 0
    held_low: np.ndarray = np.full_like(low, np.nan)
    held_high: np.ndarray = np.full_like(high, np.nan)
    held_low[held] = low[last[held]]
    held_high[held] = high[last[held]]
    return held_low, held_high
