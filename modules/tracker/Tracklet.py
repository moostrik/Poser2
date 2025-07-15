# TODO
# make sure that tracking removed from person is handled correctly

# Standard library imports
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

# Third-party imports
from pandas import Timestamp

# Local application imports
from modules.cam.depthcam.Definitions import Tracklet as ExternalTracklet
from modules.utils.HotReloadMethods import HotReloadMethods
from modules.tracker.BaseTracker import BaseTrackerInfo

class TrackingStatus(Enum):
    NEW =       0
    TRACKED =   1
    LOST =      2
    REMOVED =   3
    NONE =      4 #?

DEPTHAI_TO_TRACKINGSTATUS: dict[ExternalTracklet.TrackingStatus, TrackingStatus] = {
    ExternalTracklet.TrackingStatus.NEW: TrackingStatus.NEW,
    ExternalTracklet.TrackingStatus.TRACKED: TrackingStatus.TRACKED,
    ExternalTracklet.TrackingStatus.LOST: TrackingStatus.LOST,
    ExternalTracklet.TrackingStatus.REMOVED: TrackingStatus.REMOVED,
    # Add more if needed
}

@dataclass
class Point3f:
    """3D point with float coordinates"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

@dataclass
class Rect:
    """Rectangle defined by top-left corner and dimensions"""
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0

    @property
    def top_left(self) -> tuple[float, float]:
        return (self.x, self.y)

    @property
    def bottom_right(self) -> tuple[float, float]:
        return (self.x + self.width, self.y + self.height)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass (frozen=True)
class Tracklet:
    cam_id: int
    id: int =                   field(default=-1)

    time_stamp: Timestamp =     field(default_factory=Timestamp.now)
    created_at: Timestamp =     field(default_factory=Timestamp.now)
    last_active: Timestamp =    field(default_factory=Timestamp.now)  # Last time person was actually detected (NEW/TRACKED)

    status: TrackingStatus =    field(default=TrackingStatus.NEW)
    roi: Rect =                 field(default=Rect())

    tracker_info: Optional[BaseTrackerInfo] = field(default = None)
    _external_tracklet: Optional[ExternalTracklet] = field(default=None, repr=False)
    needs_notification: bool =  field(default=True, repr=False)

    @property
    def is_new(self) -> bool:
        return self.status == TrackingStatus.NEW

    @property
    def is_tracked(self) -> bool:
        return self.status == TrackingStatus.TRACKED

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
        return (self.last_active - self.created_at).total_seconds()

    def is_expired(self, threshold: float) -> bool:
        """Check if person hasn't been updated recently"""
        return (Timestamp.now() - self.last_active).total_seconds() > threshold

    @property
    def external_id(self) -> int:
        if self._external_tracklet:
            return self._external_tracklet.id
        return -1

    @property
    def external_age_in_frames(self)  -> int:
        if self._external_tracklet:
            return self._external_tracklet.age
        return 0

    @classmethod
    def from_depthcam(cls, cam_id: int, dct: 'ExternalTracklet') -> Optional['Tracklet']:
        """
        Initialize a Tracklet from a DepthCamTracklet instance.
        """
        status: TrackingStatus = TrackingStatus.NONE
        if hasattr(dct, 'status'):
            status = DEPTHAI_TO_TRACKINGSTATUS.get(dct.status, TrackingStatus.NONE)
        else:
            warnings.warn("Missing 'status' in DepthCamTracklet, defaulting to None.")
            return None

        roi: Optional[Rect] = None
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
            _external_tracklet=dct,
        )


# Type Aliases
TrackletCallback = Callable[[Tracklet], None]
TrackletDict = dict[int, Tracklet]
TrackletDictCallback = Callable[[TrackletDict], None]

TrackletIdColorDict: dict[int, str] = {
    0: '#006400',   # darkgreen
    1: '#00008b',   # darkblue
    2: '#b03060',   # maroon3
    3: '#ff0000',   # red
    4: '#ffff00',   # yellow
    5: '#deb887',   # burlywood
    6: '#00ff00',   # lime
    7: '#00ffff',   # aqua
    8: '#ff00ff',   # fuchsia
    9: '#6495ed',   # cornflower
}

def TrackletIdColor(id: int, aplha: float = 0.5) -> list[float]:
    hex_color: str = TrackletIdColorDict.get(id, '#000000')
    rgb: list[float] =  [int(hex_color[i:i+2], 16) / 255.0 for i in (1, 3, 5)]
    rgb.append(aplha)
    return rgb