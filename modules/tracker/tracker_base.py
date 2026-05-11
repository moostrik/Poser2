from __future__ import annotations

# Standard library imports
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

# Local application imports
from modules.oak import DepthTracklet

# Forward declaration to avoid circular import
if TYPE_CHECKING:
    from .tracklet import TrackletDictCallback


class TrackerAnnotation:
    pass


class BaseTracker(ABC):
    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def notify_update(self) -> None: ...

    @abstractmethod
    def submit_cam_tracklets(self, cam_id: int, cam_tracklets: list[DepthTracklet]) -> None: ...

    @abstractmethod
    def add_tracklet_callback(self, callback: TrackletDictCallback) -> None: ...

