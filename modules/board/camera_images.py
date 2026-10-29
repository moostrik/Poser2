from __future__ import annotations

from threading import Lock
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from modules.inference import source


class HasCameraImages(Protocol):
    """Source image access keyed by cam_id."""
    def get_camera_image(self, cam_id: int) -> "torch.Tensor | None": ...
    def set_camera_images(self, images: "source.ImageDict") -> None: ...


class CameraImageStoreMixin:
    """Thread-safe source image storage, keyed by cam_id."""

    def __init__(self) -> None:
        self._camera_image_lock = Lock()
        self._camera_images: "dict[int, torch.Tensor]" = {}

    def get_camera_image(self, cam_id: int) -> "torch.Tensor | None":
        with self._camera_image_lock:
            return self._camera_images.get(cam_id)

    def set_camera_images(self, images: "source.ImageDict") -> None:
        with self._camera_image_lock:
            self._camera_images = images
