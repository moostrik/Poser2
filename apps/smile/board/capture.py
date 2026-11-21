from __future__ import annotations

from threading import Lock
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class HasCapture(Protocol):
    """The visitor's capture (BGR, the portrait the generator starts from) and its depth (the flash out's mask)."""
    def get_capture(self) -> np.ndarray | None: ...
    def set_capture(self, image: np.ndarray) -> None: ...
    def get_capture_depth(self) -> np.ndarray | None: ...
    def set_capture_depth(self, image: np.ndarray) -> None: ...


class CaptureStoreMixin:
    """Thread-safe storage of the newest capture and its depth."""

    def __init__(self) -> None:
        self._capture_lock = Lock()
        self._capture: np.ndarray | None = None
        self._capture_depth: np.ndarray | None = None

    def get_capture(self) -> np.ndarray | None:
        with self._capture_lock:
            return self._capture

    def set_capture(self, image: np.ndarray) -> None:
        with self._capture_lock:
            self._capture = image

    def get_capture_depth(self) -> np.ndarray | None:
        with self._capture_lock:
            return self._capture_depth

    def set_capture_depth(self, image: np.ndarray) -> None:
        with self._capture_lock:
            self._capture_depth = image
