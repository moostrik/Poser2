from __future__ import annotations

from threading import Lock
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from modules.camera import CameraTracklet


class HasCameraTracklets(Protocol):
    """The tracklets each camera reports, as it reports them."""
    def get_camera_tracklets(self, cam_id: int) -> list[CameraTracklet] | None: ...
    def set_camera_tracklets(self, cam_id: int, tracklets: list[CameraTracklet], timestamp: float) -> None: ...


class CameraTrackletStoreMixin:
    """Thread-safe camera tracklet storage."""

    def __init__(self) -> None:
        self._camera_tracklet_lock = Lock()
        self._camera_tracklets: dict[int, list[CameraTracklet]] = {}

    def get_camera_tracklets(self, cam_id: int) -> list[CameraTracklet] | None:
        with self._camera_tracklet_lock:
            return self._camera_tracklets.get(cam_id)

    def set_camera_tracklets(self, cam_id: int, tracklets: list[CameraTracklet], timestamp: float) -> None:
        with self._camera_tracklet_lock:
            self._camera_tracklets[cam_id] = tracklets
