from __future__ import annotations

from enum import Enum
from typing import Protocol, List
from modules.cam.depthcam.Definitions import Tracklet as CamTracklet
# Forward declaration to avoid circular import

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from modules.tracker.Tracklet import Tracklet, TrackletCallback
# from modules.tracker.Tracklet import Tracklet, TrackletCallback

class TrackerType(Enum):
    UNKNOWN = "unknown"
    PANORAMIC = "panoramic"
    ONEPERCAM = "screenbound"

class TrackerMetadata(Protocol):
    @property
    def tracker_type(self) -> TrackerType:
        """Get the tracker type for this info"""
        ...

class BaseTracker(Protocol):
    # Core methods that all trackers must implement
    def start(self) -> None:
        """Start the tracker"""
        ...

    def stop(self) -> None:
        """Stop the tracker"""
        ...

    def add_cam_tracklets(self, cam_id: int, cam_tracklets: List[CamTracklet]) -> None:
        """Add tracklets from a camera"""
        ...

    def add_tracklet_callback(self, callback: TrackletCallback) -> None:
        """Add a callback for tracklet updates"""
        ...

    @property
    def tracker_type(self) -> TrackerType:
        """Get the tracker type"""
        ...

