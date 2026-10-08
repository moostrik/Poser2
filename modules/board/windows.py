from __future__ import annotations

from threading import Lock
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from modules.pose.frame import FeatureWindow
    from modules.pose.features import BaseFeature


class WindowSource(Protocol):
    """What a stage's window provider offers: one window on demand (a WindowTracker)."""
    def get_window(self, feature_type: type[BaseFeature], track_id: int) -> FeatureWindow | None: ...


class HasWindows(Protocol):
    """Staged feature window access."""
    def get_window(self, stage: int, track_id: int, feature_type: type[BaseFeature]) -> FeatureWindow | None: ...
    def set_window_tracker(self, stage: int, source: WindowSource) -> None: ...


class WindowStoreMixin:
    """Thread-safe staged feature window access.

    Stores one window source per stage (set once at wiring time); a read builds that one
    window on demand in the source, so no per-frame window data crosses the board.
    """

    def __init__(self) -> None:
        self._window_lock = Lock()
        self._window_sources: dict[int, WindowSource] = {}

    def get_window(self, stage: int, track_id: int, feature_type: type[BaseFeature]) -> FeatureWindow | None:
        with self._window_lock:
            source = self._window_sources.get(stage)
        return source.get_window(feature_type, track_id) if source is not None else None

    def set_window_tracker(self, stage: int, source: WindowSource) -> None:
        with self._window_lock:
            self._window_sources[stage] = source
