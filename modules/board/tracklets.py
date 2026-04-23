from __future__ import annotations

from threading import Lock
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from modules.tracker import Tracklet


class HasTracklets(Protocol):
    """Panoramic tracker tracklet access."""
    def get_tracklets(self) -> dict[int, Tracklet]: ...
    def set_tracklets(self, tracklets: dict[int, Tracklet]) -> None: ...


class TrackletStoreMixin:
    """Thread-safe panoramic tracklet storage."""

    def __init__(self) -> None:
        self._tracklet_store_lock = Lock()
        self._tracklets: dict[int, Tracklet] = {}

    def get_tracklets(self) -> dict[int, Tracklet]:
        with self._tracklet_store_lock:
            return dict(self._tracklets)

    def set_tracklets(self, tracklets: dict[int, Tracklet]) -> None:
        with self._tracklet_store_lock:
            self._tracklets = dict(tracklets)
