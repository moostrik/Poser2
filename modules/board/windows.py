from __future__ import annotations

from threading import Lock
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from modules.pose.frame import FeatureWindow, FrameWindowDict
    from modules.pose.features.base import BaseFeature


class HasWindows(Protocol):
    """Staged feature window access."""
    def get_window(self, stage: int, track_id: int, feature_type: type[BaseFeature]) -> FeatureWindow | None: ...
    def set_windows(self, stage: int, windows: FrameWindowDict) -> None: ...


class WindowStoreMixin:
    """Thread-safe staged feature window storage."""

    def __init__(self) -> None:
        self._window_lock = Lock()
        self._windows: dict[int, dict[int, dict]] = {}

    def get_window(self, stage: int, track_id: int, feature_type: type[BaseFeature]) -> FeatureWindow | None:
        with self._window_lock:
            track_windows = self._windows.get(stage, {}).get(track_id)
            if track_windows is None:
                return None
            return track_windows.get(feature_type)

    def set_windows(self, stage: int, windows: FrameWindowDict) -> None:
        pivoted: dict[int, dict] = {}
        for feature, track_windows in windows.items():
            for track_id, window in track_windows.items():
                if track_id not in pivoted:
                    pivoted[track_id] = {}
                pivoted[track_id][feature] = window
        with self._window_lock:
            self._windows[stage] = pivoted
