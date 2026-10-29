from __future__ import annotations

from threading import Lock
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from modules.inference import crop


class HasCropImages(Protocol):
    """Crop image access keyed by track_id."""
    def get_crop_image(self, track_id: int) -> "crop.Image | None": ...
    def set_crop_images(self, images: "crop.CropImageDict") -> None: ...


class CropImageStoreMixin:
    """Thread-safe crop image storage, keyed by track_id."""

    def __init__(self) -> None:
        self._crop_image_lock = Lock()
        self._crop_images: "dict[int, crop.Image]" = {}

    def get_crop_image(self, track_id: int) -> "crop.Image | None":
        with self._crop_image_lock:
            return self._crop_images.get(track_id)

    def set_crop_images(self, images: "crop.CropImageDict") -> None:
        with self._crop_image_lock:
            self._crop_images = images
