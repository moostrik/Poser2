from __future__ import annotations

from collections import deque
from threading import Lock
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from modules.inference import Capture

KEPT: int = 4       # captures a camera keeps for the render's next tick; beyond that the oldest is dropped and counted


class HasCaptures(Protocol):
    """Each camera's captures since the render took them last: color, depth and poses of one moment, keyed by cam_id."""
    def take_captures(self, cam_id: int) -> "list[Capture]": ...
    def set_capture(self, capture: "Capture") -> None: ...
    def get_dropped_captures(self) -> int: ...


class CaptureStoreMixin:
    """Thread-safe capture storage: each camera's captures in the order they came, until the render takes them.
    A reader takes what is there and never waits for it. Beyond `KEPT` waiting the oldest is dropped and counted."""

    def __init__(self) -> None:
        self._capture_lock = Lock()
        self._captures: "dict[int, deque[Capture]]" = {}
        self._captures_dropped: int = 0

    def take_captures(self, cam_id: int) -> "list[Capture]":
        """The camera's captures since the last take, oldest first."""
        with self._capture_lock:
            waiting: "deque[Capture] | None" = self._captures.get(cam_id)
            if not waiting:
                return []
            captures: "list[Capture]" = list(waiting)
            waiting.clear()
            return captures

    def set_capture(self, capture: "Capture") -> None:
        with self._capture_lock:
            waiting: "deque[Capture]" = self._captures.setdefault(capture.cam_id, deque())
            if len(waiting) >= KEPT:
                waiting.popleft()
                self._captures_dropped += 1
            waiting.append(capture)

    def get_dropped_captures(self) -> int:
        """Captures dropped before the render took them."""
        with self._capture_lock:
            return self._captures_dropped
