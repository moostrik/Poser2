from __future__ import annotations

from threading import Lock
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from modules.oak.camera.definitions import Tracklet


class HasDepthTracklets(Protocol):
    """Depth tracklet access."""
    def get_depth_tracklets(self, cam_id: int) -> list[Tracklet] | None: ...
    def set_depth_tracklets(self, cam_id: int, tracklets: list[Tracklet]) -> None: ...


class DepthTrackletStoreMixin:
    """Thread-safe depth tracklet storage."""

    def __init__(self) -> None:
        self._tracklet_lock = Lock()
        self._depth_tracklets: dict[int, list[Tracklet]] = {}

    def get_depth_tracklets(self, cam_id: int) -> list[Tracklet] | None:
        with self._tracklet_lock:
            return self._depth_tracklets.get(cam_id)

    def set_depth_tracklets(self, cam_id: int, tracklets: list[Tracklet]) -> None:
        with self._tracklet_lock:
            self._depth_tracklets[cam_id] = tracklets
