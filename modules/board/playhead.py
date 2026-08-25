from __future__ import annotations

from threading import Lock
from typing import Protocol


class HasPlayhead(Protocol):
    """Rotating-light playhead access (radians [-π, π); NaN when not meaningful)."""
    def get_playhead(self) -> float: ...
    def set_playhead(self, playhead: float) -> None: ...


class PlayheadStoreMixin:
    """Thread-safe playhead storage. NaN means the playhead is not currently meaningful."""

    def __init__(self) -> None:
        self._playhead_lock = Lock()
        self._playhead: float = float("nan")

    def get_playhead(self) -> float:
        with self._playhead_lock:
            return self._playhead

    def set_playhead(self, playhead: float) -> None:
        with self._playhead_lock:
            self._playhead = playhead
