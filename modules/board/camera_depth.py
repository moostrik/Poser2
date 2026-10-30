from __future__ import annotations

from threading import Lock
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from modules.camera import CameraFrame


class HasCameraDepth(Protocol):
    """Each camera's newest DEPTH frame, keyed by cam_id."""
    def get_camera_depth(self, cam_id: int) -> "CameraFrame | None": ...
    def set_camera_depth(self, frame: "CameraFrame") -> None: ...


class CameraDepthStoreMixin:
    """Thread-safe depth storage, the newest frame of each camera. A reader takes what is there and
    never waits for it, so the color frames keep their own timing."""

    def __init__(self) -> None:
        self._camera_depth_lock = Lock()
        self._camera_depths: "dict[int, CameraFrame]" = {}

    def get_camera_depth(self, cam_id: int) -> "CameraFrame | None":
        with self._camera_depth_lock:
            return self._camera_depths.get(cam_id)

    def set_camera_depth(self, frame: "CameraFrame") -> None:
        with self._camera_depth_lock:
            self._camera_depths[frame.cam_id] = frame
