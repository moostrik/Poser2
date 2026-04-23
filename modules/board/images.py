from __future__ import annotations

from threading import Lock
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from modules.pose.batch.CameraImage import CameraImage, CameraImageDict
    from modules.pose.batch.CropImage import CropImage, CropImageDict
    from modules.pose.batch.MaskImage import MaskImage, MaskImageDict


class HasCameraImages(Protocol):
    """Full camera image access keyed by cam_id."""
    def get_camera_image(self, cam_id: int) -> CameraImage | None: ...
    def set_camera_images(self, images: CameraImageDict) -> None: ...


class CameraImageStoreMixin:
    """Thread-safe full camera image storage, keyed by cam_id."""

    def __init__(self) -> None:
        self._cam_image_lock = Lock()
        self._cam_images: dict[int, CameraImage] = {}

    def get_camera_image(self, cam_id: int) -> CameraImage | None:
        with self._cam_image_lock:
            return self._cam_images.get(cam_id)

    def set_camera_images(self, images: CameraImageDict) -> None:
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


class HasMaskImages(Protocol):
    """Mask image access keyed by track_id."""
    def get_mask_image(self, track_id: int) -> MaskImage | None: ...
    def set_mask_images(self, images: MaskImageDict) -> None: ...


class MaskImageStoreMixin:
    """Thread-safe mask image storage, keyed by track_id."""

    def __init__(self) -> None:
        self._mask_image_lock = Lock()
        self._mask_images: dict[int, MaskImage] = {}

    def get_mask_image(self, track_id: int) -> MaskImage | None:
        with self._mask_image_lock:
            return self._mask_images.get(track_id)

    def set_mask_images(self, images: MaskImageDict) -> None:
        with self._mask_image_lock:
            self._mask_images = images
