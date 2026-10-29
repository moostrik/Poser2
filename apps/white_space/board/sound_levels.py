from __future__ import annotations

from dataclasses import dataclass, replace
from threading import Lock
from time import perf_counter
from typing import Protocol


@dataclass(frozen=True, slots=True)
class SoundLevels:
    """Sound levels received from the audio side (pure data; producers define the scale —
    for white_space: two 0..1 channel levels from Max via ``/WS/idle/blue/left`` and
    ``/WS/idle/blue/right``).
    ``timestamp`` is the receive time on perf_counter (0.0 = never received) so consumers can
    detect stale input."""
    left:      float = 0.0
    right:     float = 0.0
    timestamp: float = 0.0


class HasSoundLevels(Protocol):
    """Audio-level access."""
    def get_sound_levels(self) -> SoundLevels: ...
    def set_sound_level_left(self, level: float) -> None: ...
    def set_sound_level_right(self, level: float) -> None: ...


class SoundLevelStoreMixin:
    """Thread-safe sound-level storage; stamps the receive time (perf_counter) on set."""

    def __init__(self) -> None:
        self._sound_level_lock = Lock()
        self._sound_levels = SoundLevels()

    def get_sound_levels(self) -> SoundLevels:
        with self._sound_level_lock:
            return self._sound_levels

    def set_sound_level_left(self, level: float) -> None:
        with self._sound_level_lock:
            self._sound_levels = replace(self._sound_levels, left=level, timestamp=perf_counter())

    def set_sound_level_right(self, level: float) -> None:
        with self._sound_level_lock:
            self._sound_levels = replace(self._sound_levels, right=level, timestamp=perf_counter())
