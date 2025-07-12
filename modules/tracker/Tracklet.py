# Standard library imports
from dataclasses import dataclass, field
from time import time
from typing import Callable

# Third-party imports
from pandas import Timestamp

# Local application imports
from modules.cam.depthcam.Definitions import Tracklet as CamTracklet
from modules.tracker.BaseTracker import BaseTrackerInfo, TrackingStatus

@dataclass
class Tracklet:
    id: int
    cam_id: int
    external_tracklet: CamTracklet
    time_stamp: Timestamp
    tracker_info: BaseTrackerInfo
    status: TrackingStatus = field(default=TrackingStatus.NONE)
    start_time: float = field(default_factory=time)
    last_time: float = field(default_factory=time)

    def __post_init__(self):
        # If you want to set status from tracklet, do it here
        if self.status == TrackingStatus.NONE and self.external_tracklet is not None:
            self.status = TrackingStatus[self.external_tracklet.status.name]

    @property
    def is_active(self) -> bool:
        return self.status in (TrackingStatus.NEW, TrackingStatus.TRACKED)

    @property
    def age(self) -> float:
        """Get how long this tracket has been tracked"""
        return time() - self.start_time

    def is_expired(self, threshold) -> bool:
        """Check if tracket hasn't been updated recently"""
        return time() - self.last_time > threshold


# Type Aliases
TrackletCallback = Callable[[Tracklet], None]
TrackletDict = dict[int, Tracklet]
TrackletDictCallback = Callable[[TrackletDict], None]


PersonColors: dict[int, str] = {
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

def PersonColor(id: int, aplha: float = 0.5) -> list[float]:
    hex_color: str = PersonColors.get(id, '#000000')
    rgb: list[float] =  [int(hex_color[i:i+2], 16) / 255.0 for i in (1, 3, 5)]
    rgb.append(aplha)
    return rgb