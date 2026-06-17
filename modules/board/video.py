from __future__ import annotations

from threading import Lock
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class HasVideoImages(Protocol):
    """Raw camera VIDEO frame access keyed by cam_id (CPU numpy frames)."""
    def get_video_image(self, cam_id: int) -> "np.ndarray | None": ...
    def set_video_image(self, cam_id: int, frame: "np.ndarray") -> None: ...


class VideoImageStoreMixin:
    """Thread-safe raw VIDEO frame storage, keyed by cam_id.

    Distinct from CameraImageStoreMixin, which holds torch inference tensors.
    This slot holds the raw CPU numpy VIDEO frames used for calibration display.
    """

    def __init__(self) -> None:
        self._video_image_lock = Lock()
        self._video_images: "dict[int, np.ndarray]" = {}

    def get_video_image(self, cam_id: int) -> "np.ndarray | None":
        with self._video_image_lock:
            return self._video_images.get(cam_id)

    def set_video_image(self, cam_id: int, frame: "np.ndarray") -> None:
        with self._video_image_lock:
            self._video_images[cam_id] = frame
