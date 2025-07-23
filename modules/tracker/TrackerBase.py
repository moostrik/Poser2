from __future__ import annotations

# Standard library imports
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

# Local application imports
from modules.cam.depthcam.Definitions import Tracklet as DepthTracklet

# Forward declaration to avoid circular import
if TYPE_CHECKING:
    from modules.tracker.Tracklet import TrackletCallback

class TrackerType(Enum):
    UNKNOWN = "unknown"
    PANORAMIC = "panoramic"
    ONEPERCAM = "onepercam"

class TrackerMetadata(ABC):
    @abstractmethod
    def tracker_type(self) -> TrackerType: ...

class BaseTracker(ABC):
    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def add_cam_tracklets(self, cam_id: int, cam_tracklets: list[DepthTracklet]) -> None: ...

    @abstractmethod
    def add_tracklet_callback(self, callback: TrackletCallback) -> None: ...

    @abstractmethod
    def tracker_type(self) -> TrackerType: ...

