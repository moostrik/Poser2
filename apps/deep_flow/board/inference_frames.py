from __future__ import annotations

from collections import deque
from threading import Lock
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from modules.inference import InferenceFrame

KEPT: int = 4       # frames a camera keeps for the render's next tick; beyond that the oldest is dropped and counted


class HasInferenceFrames(Protocol):
    """Each camera's frames since the render took them last: color, depth and poses of one moment, keyed by cam_id."""
    def take_inference_frames(self, cam_id: int) -> "list[InferenceFrame]": ...
    def set_inference_frame(self, frame: "InferenceFrame") -> None: ...
    def get_dropped_inference_frames(self) -> int: ...


class InferenceFrameStoreMixin:
    """Thread-safe frame storage: each camera's frames in the order they came, until the render takes them.
    A reader takes what is there and never waits for it. Beyond `KEPT` waiting the oldest is dropped and counted."""

    def __init__(self) -> None:
        self._inference_frame_lock = Lock()
        self._inference_frames: "dict[int, deque[InferenceFrame]]" = {}
        self._inference_frames_dropped: int = 0

    def take_inference_frames(self, cam_id: int) -> "list[InferenceFrame]":
        """The camera's frames since the last take, oldest first."""
        with self._inference_frame_lock:
            waiting: "deque[InferenceFrame] | None" = self._inference_frames.get(cam_id)
            if not waiting:
                return []
            frames: "list[InferenceFrame]" = list(waiting)
            waiting.clear()
            return frames

    def set_inference_frame(self, frame: "InferenceFrame") -> None:
        with self._inference_frame_lock:
            waiting: "deque[InferenceFrame]" = self._inference_frames.setdefault(frame.cam_id, deque())
            if len(waiting) >= KEPT:
                waiting.popleft()
                self._inference_frames_dropped += 1
            waiting.append(frame)

    def get_dropped_inference_frames(self) -> int:
        """Inference frames dropped before the render took them."""
        with self._inference_frame_lock:
            return self._inference_frames_dropped
