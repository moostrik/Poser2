from __future__ import annotations

from threading import Lock
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from modules.inference import Capture


class HasCaptures(Protocol):
    """Each camera's newest capture: color, depth and poses of one moment, keyed by cam_id."""
    def get_capture(self, cam_id: int) -> "Capture | None": ...
    def set_capture(self, capture: "Capture") -> None: ...


class CaptureStoreMixin:
    """Thread-safe capture storage, the newest capture of each camera. A reader takes what is there
    and never waits for it."""

    def __init__(self) -> None:
        self._capture_lock = Lock()
        self._captures: "dict[int, Capture]" = {}

    def get_capture(self, cam_id: int) -> "Capture | None":
        with self._capture_lock:
            return self._captures.get(cam_id)

    def set_capture(self, capture: "Capture") -> None:
        with self._capture_lock:
            self._captures[capture.cam_id] = capture
