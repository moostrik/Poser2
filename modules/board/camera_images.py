from __future__ import annotations

from threading import Lock
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from modules.inference import GpuImage


class HasCameraImages(Protocol):
    """Each camera's newest uploaded image, with the event that says it is written."""
    def get_camera_image(self, cam_id: int) -> "GpuImage | None": ...
    def set_camera_image(self, image: "GpuImage") -> None: ...


class CameraImageStoreMixin:
    """Thread-safe storage of each camera's newest uploaded image, keyed by cam_id."""

    def __init__(self) -> None:
        self._camera_image_lock = Lock()
        self._camera_images: "dict[int, GpuImage]" = {}

    def get_camera_image(self, cam_id: int) -> "GpuImage | None":
        with self._camera_image_lock:
            return self._camera_images.get(cam_id)

    def set_camera_image(self, image: "GpuImage") -> None:
        with self._camera_image_lock:
            self._camera_images[image.cam_id] = image
