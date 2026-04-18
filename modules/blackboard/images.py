from __future__ import annotations

from threading import Lock
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from modules.pose.batch.ImageFrame import ImageFrame


class HasImages(Protocol):
    """GPU image frame access."""
    def get_image(self, track_id: int) -> ImageFrame | None: ...
    def set_images(self, frames: dict[int, ImageFrame]) -> None: ...


class ImageStoreMixin:
    """Thread-safe GPU image frame storage."""

    def __init__(self) -> None:
        self._gpu_frame_lock = Lock()
        self._gpu_frames: dict[int, ImageFrame] = {}

    def get_image(self, track_id: int) -> ImageFrame | None:
        with self._gpu_frame_lock:
            return self._gpu_frames.get(track_id)

    def set_images(self, frames: dict[int, ImageFrame]) -> None:
        with self._gpu_frame_lock:
            self._gpu_frames = frames
