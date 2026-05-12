# Standard library imports
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, TypeAlias

# Local application imports
from modules.oak import DepthTracklet
from modules.utils import Rect
from .tracker_base import TrackerAnnotation


class TrackingStatus(Enum):
    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3
    NONE = 4  # ?


DEPTHAI_TO_TRACKINGSTATUS: dict[DepthTracklet.TrackingStatus, TrackingStatus] = {
    DepthTracklet.TrackingStatus.NEW: TrackingStatus.NEW,
    DepthTracklet.TrackingStatus.TRACKED: TrackingStatus.TRACKED,
    DepthTracklet.TrackingStatus.LOST: TrackingStatus.LOST,
    DepthTracklet.TrackingStatus.REMOVED: TrackingStatus.REMOVED,
}


@dataclass(frozen=True)
class Tracklet:
    cam_id: int
    id: int = field(default=-1)

    time_stamp: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)  # Last time person was actually detected (NEW/TRACKED)

    status: TrackingStatus = field(default=TrackingStatus.NEW)
    roi: Rect = field(default_factory=Rect)

    annotation: TrackerAnnotation | None = field(default=None)
    external_id: int = field(default=-1)
    external_age_in_frames: int = field(default=0)

    @property
    def is_lost(self) -> bool:
        return self.status == TrackingStatus.LOST

    @property
    def is_removed(self) -> bool:
        return self.status == TrackingStatus.REMOVED

    @property
    def is_active(self) -> bool:
        return self.status in (TrackingStatus.NEW, TrackingStatus.TRACKED)

    @property
    def age_in_seconds(self) -> float:
        """Get how long this person has been tracked"""
        return self.last_active - self.created_at

    def is_expired(self, threshold: float) -> bool:
        """Returns True if the tracklet has not been updated within threshold seconds."""
        return (time.time() - self.last_active) > threshold

    @classmethod
    def from_depthcam(cls, cam_id: int, dct: 'DepthTracklet') -> 'Tracklet | None':
        """
        Initialize a Tracklet from a DepthCamTracklet instance.
        """
        status: TrackingStatus = TrackingStatus.NONE
        if hasattr(dct, 'status'):
            status = DEPTHAI_TO_TRACKINGSTATUS.get(dct.status, TrackingStatus.NONE)
        else:
            warnings.warn("Missing 'status' in DepthCamTracklet, defaulting to None.")
            return None

        roi: Rect | None = None
        if hasattr(dct, 'roi'):
            roi = Rect(
                x=dct.roi.x,
                y=dct.roi.y,
                width=dct.roi.width,
                height=dct.roi.height
            )
        else:
            warnings.warn("Missing 'roi' in DepthCamTracklet, setting to None.")
            return None

        return cls(
            cam_id=cam_id,
            status=status,
            roi=roi,
            external_id=dct.id,
            external_age_in_frames=dct.age,
        )



# Type Aliases
TrackletCallback: TypeAlias = Callable[[Tracklet], None]
TrackletDict: TypeAlias = dict[int, Tracklet]
TrackletDictCallback: TypeAlias = Callable[[TrackletDict], None]
