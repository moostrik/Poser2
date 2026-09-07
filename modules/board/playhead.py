from __future__ import annotations

from threading import Lock
from typing import Protocol


class HasPlayhead(Protocol):
    """Rotating-light playhead access: phase (radians [-π, π); NaN when not meaningful)
    and the monotonic bar counter (1 bar = 1 full playhead cycle; never NaN)."""
    def get_playhead(self) -> float: ...
    def get_playhead_bars(self) -> float: ...
    def set_playhead(self, playhead: float, bars: float = 0.0) -> None: ...


class PlayheadStoreMixin:
    """Thread-safe playhead storage. NaN phase means the playhead is not currently meaningful;
    `bars` is the monotonic content clock and is always finite."""

    def __init__(self) -> None:
        self._playhead_lock = Lock()
        self._playhead: float = float("nan")
        self._playhead_bars: float = 0.0

    def get_playhead(self) -> float:
        with self._playhead_lock:
            return self._playhead

    def get_playhead_bars(self) -> float:
        with self._playhead_lock:
            return self._playhead_bars

    def set_playhead(self, playhead: float, bars: float = 0.0) -> None:
        with self._playhead_lock:
            self._playhead = playhead
            self._playhead_bars = bars
