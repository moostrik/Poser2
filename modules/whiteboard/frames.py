from __future__ import annotations

from threading import Lock
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from modules.pose.frame import Frame


class HasFrames(Protocol):
    """Staged pose frame access."""
    def get_frame(self, stage: int, track_id: int) -> Frame | None: ...
    def get_frames(self, stage: int) -> dict[int, Frame]: ...
    def set_frames(self, stage: int, frames: dict[int, Frame]) -> None: ...


class FrameStoreMixin:
    """Thread-safe staged frame storage."""

    def __init__(self) -> None:
        self._frame_lock = Lock()
        self._frames: dict[int, dict[int, Frame]] = {}

    def get_frame(self, stage: int, track_id: int) -> Frame | None:
        with self._frame_lock:
            return self._frames.get(stage, {}).get(track_id)

    def get_frames(self, stage: int) -> dict[int, Frame]:
        with self._frame_lock:
            return dict(self._frames.get(stage, {}))

    def set_frames(self, stage: int, frames: dict[int, Frame]) -> None:
        with self._frame_lock:
            self._frames[stage] = dict(frames)
