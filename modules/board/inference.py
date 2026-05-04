from __future__ import annotations

from threading import Lock
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from modules.inference import source, crop, segmentation
    # from modules.inference.segmentation import Image as SegmentationImage, ImageDict as SegmentationImageDict


class HasCameraImages(Protocol):
    """Source image access keyed by cam_id."""
    def get_camera_image(self, cam_id: int) -> "torch.Tensor | None": ...
    def set_camera_images(self, images: "source.ImageDict") -> None: ...


class CameraImageStoreMixin:
    """Thread-safe source image storage, keyed by cam_id."""

    def __init__(self) -> None:
        self._cam_image_lock = Lock()
        self._cam_images: "dict[int, torch.Tensor]" = {}

    def get_camera_image(self, cam_id: int) -> "torch.Tensor | None":
        with self._cam_image_lock:
            return self._cam_images.get(cam_id)

    def set_camera_images(self, images: "source.ImageDict") -> None:
        with self._cam_image_lock:
            self._cam_images = images


class HasCropImages(Protocol):
    """Crop image access keyed by track_id."""
    def get_crop_image(self, track_id: int) -> "crop.Image | None": ...
    def set_crop_images(self, images: "crop.ImageDict") -> None: ...


class CropImageStoreMixin:
    """Thread-safe crop image storage, keyed by track_id."""

    def __init__(self) -> None:
        self._crop_image_lock = Lock()
        self._crop_images: "dict[int, crop.Image]" = {}

    def get_crop_image(self, track_id: int) -> "crop.Image | None":
        with self._crop_image_lock:
            return self._crop_images.get(track_id)

    def set_crop_images(self, images: "crop.ImageDict") -> None:
        with self._crop_image_lock:
            self._crop_images = images


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
