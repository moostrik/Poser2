from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import Lock
from time import monotonic
from typing import Protocol

# Recent flashes kept for readers; consumers select by age, so this only bounds memory.
_FLASH_CAPACITY: int = 64


@dataclass(frozen=True, slots=True)
class Flash:
    """One tick of a light flash (pure data; producers define the meaning — for white_space: the
    bar's heading in radians while the flash lit it, and the white and blue flash levels 0..1).
    ``timestamp`` is the monotonic time it was added."""
    azimuth:   float
    white:     float
    blue:      float
    timestamp: float


class HasFlashes(Protocol):
    """Recent-flash access."""
    def add_flash(self, azimuth: float, white: float, blue: float) -> None: ...
    def get_flashes(self) -> list[Flash]: ...


class FlashStoreMixin:
    """Thread-safe store of the most recent flashes, oldest first; stamps the monotonic time on add."""

    def __init__(self) -> None:
        self._flash_lock = Lock()
        self._flashes: deque[Flash] = deque(maxlen=_FLASH_CAPACITY)

    def add_flash(self, azimuth: float, white: float, blue: float) -> None:
        with self._flash_lock:
            self._flashes.append(Flash(azimuth, white, blue, monotonic()))

    def get_flashes(self) -> list[Flash]:
        with self._flash_lock:
            return list(self._flashes)
