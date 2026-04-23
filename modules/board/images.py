from __future__ import annotations

from threading import Lock
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from modules.inference.image_uploader import FullImage, FullImageDict
    from modules.inference.crop_extractor import CropImage, CropImageDict
    from modules.inference.segmentation.segmentation_image import SegmentationImage, SegmentationImageDict


class HasCameraImages(Protocol):
    """Full camera image access keyed by cam_id."""
    def get_camera_image(self, cam_id: int) -> FullImage | None: ...
    def set_camera_images(self, images: FullImageDict) -> None: ...


class CameraImageStoreMixin:
    """Thread-safe full camera image storage, keyed by cam_id."""

    def __init__(self) -> None:
        self._cam_image_lock = Lock()
        self._cam_images: dict[int, FullImage] = {}

    def get_camera_image(self, cam_id: int) -> FullImage | None:
        with self._cam_image_lock:
            return self._cam_images.get(cam_id)

    def set_camera_images(self, images: FullImageDict) -> None:
        with self._cam_image_lock:
            self._cam_images = images


class HasCropImages(Protocol):
    """Crop image access keyed by track_id."""
    def get_crop_image(self, track_id: int) -> CropImage | None: ...
    def set_crop_images(self, images: CropImageDict) -> None: ...


class CropImageStoreMixin:
    """Thread-safe crop image storage, keyed by track_id."""

    def __init__(self) -> None:
        self._crop_image_lock = Lock()
        self._crop_images: dict[int, CropImage] = {}

    def get_crop_image(self, track_id: int) -> CropImage | None:
        with self._crop_image_lock:
            return self._crop_images.get(track_id)

    def set_crop_images(self, images: CropImageDict) -> None:
        with self._crop_image_lock:
            self._crop_images = images


class HasSegmentationImages(Protocol):
    """Segmentation image access keyed by track_id."""
    def get_segmentation_image(self, track_id: int) -> SegmentationImage | None: ...
    def set_segmentation_images(self, images: SegmentationImageDict) -> None: ...


class SegmentationImageStoreMixin:
    """Thread-safe segmentation image storage, keyed by track_id."""

    def __init__(self) -> None:
        self._segmentation_image_lock = Lock()
        self._segmentation_images: dict[int, SegmentationImage] = {}

    def get_segmentation_image(self, track_id: int) -> SegmentationImage | None:
        with self._segmentation_image_lock:
            return self._segmentation_images.get(track_id)

    def set_segmentation_images(self, images: SegmentationImageDict) -> None:
        with self._segmentation_image_lock:
            self._segmentation_images = images
