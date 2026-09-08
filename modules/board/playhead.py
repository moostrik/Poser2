from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Protocol


@dataclass(frozen=True, slots=True)
class PlayheadSignals:
    """The playhead's published clock + regime signals (pure data; producers define the
    semantics — for white_space: phase in radians [-π, π) or NaN, bars = monotonic
    content-clock cycles, synced = tracking the measured rotation at content speed,
    ring_formed = the bar has physically blurred into the POV ring)."""
    phase:       float = float("nan")
    bars:        float = 0.0
    synced:      bool  = False
    ring_formed: bool  = False


class HasPlayhead(Protocol):
    """Rotating-light playhead access: the phase/bars shortcuts plus the full signal bundle."""
    def get_playhead(self) -> float: ...
    def get_playhead_bars(self) -> float: ...
    def get_playhead_signals(self) -> PlayheadSignals: ...
    def set_playhead(self, signals: PlayheadSignals) -> None: ...


class PlayheadStoreMixin:
    """Thread-safe playhead storage. NaN phase means the playhead is not currently
    meaningful; `bars` is the monotonic content clock and is always finite."""

    def __init__(self) -> None:
        self._playhead_lock = Lock()
        self._playhead_signals = PlayheadSignals()

    def get_playhead(self) -> float:
        with self._playhead_lock:
            return self._playhead_signals.phase

    def get_playhead_bars(self) -> float:
        with self._playhead_lock:
            return self._playhead_signals.bars

    def get_playhead_signals(self) -> PlayheadSignals:
        with self._playhead_lock:
            return self._playhead_signals

    def set_playhead(self, signals: PlayheadSignals) -> None:
        with self._playhead_lock:
            self._playhead_signals = signals
