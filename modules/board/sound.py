from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from time import monotonic
from typing import Protocol


@dataclass(frozen=True, slots=True)
class SoundLevels:
    """Sound levels received from the audio side (pure data; producers define the scale —
    for white_space: two 0..1 channel levels from Max via ``/WS/sound/level``).
    ``timestamp`` is the monotonic receive time (0.0 = never received) so consumers can
    detect stale input."""
    left:      float = 0.0
    right:     float = 0.0
    timestamp: float = 0.0


class HasSoundLevels(Protocol):
    """Audio-level access."""
    def get_sound_levels(self) -> SoundLevels: ...
    def set_sound_levels(self, left: float, right: float) -> None: ...


class SoundLevelStoreMixin:
    """Thread-safe sound-level storage; stamps the monotonic receive time on set."""

    def __init__(self) -> None:
        self._sound_lock = Lock()
        self._sound_levels = SoundLevels()

    def get_sound_levels(self) -> SoundLevels:
        with self._sound_lock:
            return self._sound_levels

    def set_sound_levels(self, left: float, right: float) -> None:
        with self._sound_lock:
            self._sound_levels = SoundLevels(left=left, right=right, timestamp=monotonic())
