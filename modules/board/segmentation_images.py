from __future__ import annotations

from threading import Lock
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from modules.inference import segmentation


class HasSegmentationImages(Protocol):
    """Segmentation image access keyed by track_id."""
    def get_segmentation_image(self, track_id: int) -> segmentation.Image | None: ...
    def set_segmentation_images(self, images: segmentation.ImageDict) -> None: ...


class SegmentationImageStoreMixin:
    """Thread-safe segmentation image storage, keyed by track_id."""

    def __init__(self) -> None:
        self._segmentation_image_lock = Lock()
        self._segmentation_images: dict[int, segmentation.Image] = {}

    def get_segmentation_image(self, track_id: int) -> segmentation.Image | None:
        with self._segmentation_image_lock:
            return self._segmentation_images.get(track_id)

    def set_segmentation_images(self, images: segmentation.ImageDict) -> None:
        with self._segmentation_image_lock:
            self._segmentation_images = images
