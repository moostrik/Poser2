# Standard library imports
from dataclasses import dataclass, field
from time import time
from threading import Lock
from typing import Optional, Callable

# Third-party imports
import cv2
import numpy as np
from pandas import Timestamp

# Local application imports
from modules.cam.depthcam.Definitions import Tracklet, Rect
from modules.pose.PoseDefinitions import Pose, JointAngleDict
from modules.person.trackers.BaseTracker import BaseTrackerInfo, TrackingStatus

@dataclass
class Person:
    id: int
    cam_id: int
    tracklet: Tracklet
    time_stamp: Timestamp
    tracker_info: BaseTrackerInfo
    status: TrackingStatus = field(default=TrackingStatus.NONE)
    start_time: float = field(default_factory=time)
    last_time: float = field(default_factory=time)

    _pose_lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _pose_crop_rect: Optional[Rect] = field(default=None, init=False, repr=False)
    _pose_image: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _pose: Optional[Pose] = field(default=None, init=False, repr=False)
    _pose_angles: Optional[JointAngleDict] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        # If you want to set status from tracklet, do it here
        if self.status == TrackingStatus.NONE and self.tracklet is not None:
            self.status = TrackingStatus[self.tracklet.status.name]

    @property
    def pose_crop_rect(self) -> Optional[Rect]:
        with self._pose_lock:
            return self._pose_crop_rect

    @pose_crop_rect.setter
    def pose_crop_rect(self, value: Optional[Rect]) -> None:
        with self._pose_lock:
            self._pose_crop_rect = value

    @property
    def pose_image(self) -> Optional[np.ndarray]:
        with self._pose_lock:
            return self._pose_image

    @pose_image.setter
    def pose_image(self, value: Optional[np.ndarray]) -> None:
        with self._pose_lock:
            self._pose_image = value

    @property
    def pose(self) -> Optional[Pose]:
        with self._pose_lock:
            return self._pose

    @pose.setter
    def pose(self, value: Optional[Pose]) -> None:
        with self._pose_lock:
            self._pose = value

    @property
    def pose_angles(self) -> Optional[JointAngleDict]:
        with self._pose_lock:
            return self._pose_angles

    @pose_angles.setter
    def pose_angles(self, value: Optional[JointAngleDict]) -> None:
        with self._pose_lock:
            self._pose_angles = value

    @property
    def is_active(self) -> bool:
        return self.status in (TrackingStatus.NEW, TrackingStatus.TRACKED)

    @property
    def age(self) -> float:
        """Get how long this person has been tracked"""
        return time() - self.start_time

    def is_expired(self, threshold) -> bool:
        """Check if person hasn't been updated recently"""
        return time() - self.last_time > threshold


# Type Aliases
PersonCallback = Callable[[Person], None]
PersonDict = dict[int, Person]
PersonDictCallback = Callable[[PersonDict], None]


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