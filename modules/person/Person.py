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

    def set_image_and_roi(self, image: np.ndarray, roi_expansion: float, output_size: int = 256) -> None:
        """
        Sets both the pose ROI and the pose image for this person.
        """
        if self.pose_crop_rect is not None or self.pose_image is not None:
            print(f"Warning: pose ROI or image already set for person {self.id} in camera {self.cam_id}.")
            return

        h, w = image.shape[:2]
        roi = self.get_crop_rect(w, h, self.tracklet.roi, roi_expansion)
        self.pose_crop_rect = roi
        self.pose_image = self.get_cropped_image(image, roi, output_size)

    @staticmethod
    def get_crop_rect(image_width: int, image_height: int, roi: Rect, expansion: float = 0.0) -> Rect:
        # Calculate the original ROI coordinates
        img_x = int(roi.x * image_width)
        img_y = int(roi.y * image_height)
        img_w = int(roi.width * image_width)
        img_h = int(roi.height * image_height)

        # Determine the size of the square cutout based on the longest side of the ROI
        img_wh: int = max(img_w, img_h)
        img_wh += int(img_wh * expansion)

        # Calculate the new coordinates to center the square cutout around the original ROI
        crop_center_x: int = img_x + img_w // 2
        crop_center_y: int = img_y + img_h // 2
        crop_x: int = crop_center_x - img_wh // 2
        crop_y: int = crop_center_y - img_wh // 2
        crop_w: int = img_wh
        crop_h: int = img_wh

        # convert back to normalized coordinates
        norm_x: float = crop_x / image_width
        norm_y: float = crop_y / image_height
        norm_w: float = crop_w / image_width
        norm_h: float = crop_h / image_height

        return Rect(norm_x, norm_y, norm_w, norm_h)

    @staticmethod
    def get_cropped_image(image: np.ndarray, roi: Rect, output_size: int) -> np.ndarray:
        image_height, image_width = image.shape[:2]
        image_channels = image.shape[2] if len(image.shape) > 2 else 1

        # Calculate the original ROI coordinates
        x: int = int(roi.x * image_width)
        y: int = int(roi.y * image_height)
        w: int = int(roi.width * image_width)
        h: int = int(roi.height * image_height)

        # Extract the roi without padding
        img_x: int = max(0, x)
        img_y: int = max(0, y)
        img_w: int = min(w + min(0, x), image_width - img_x)
        img_h: int = min(h + min(0, y), image_height - img_y)

        crop: np.ndarray = image[img_y:img_y + img_h, img_x:img_x + img_w]

        if image_channels == 1:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)

        # Apply padding if the roi is outside the image bounds
        left_padding: int = -min(0, x)
        top_padding: int = -min(0, y)
        right_padding: int = max(0, x + w - image_width)
        bottom_padding: int = max(0, y + h - image_height)

        if left_padding + right_padding + top_padding + bottom_padding > 0:
            crop = cv2.copyMakeBorder(crop, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        # Resize the cutout to the desired size
        return cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA)

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