import cv2
import numpy as np
from time import time

from typing import Callable
from modules.cam.depthcam.Definitions import Tracklet, Rect
from modules.pose.Definitions import PoseList


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


class Person():
    def __init__(self, id, cam_id: int, tracklet: Tracklet) -> None:
        self.id: int =                  id
        self.cam_id: int =              cam_id
        self.tracklet: Tracklet =       tracklet

        self.local_angle: float =       0.0
        self.world_angle: float =       0.0
        self.pose: PoseList | None =    None
        self.pose_roi: Rect | None =    None
        self.img: np.ndarray | None =   None

        self.active: bool =             True
        self.start_time: float =        time()
        self.last_time: float =         time()

        self.overlap: bool =            False

    def update_from(self, other: 'Person') -> None:
        # self.id = other.id
        self.cam_id = other.cam_id
        self.tracklet = other.tracklet
        self.local_angle = other.local_angle
        self.world_angle = other.world_angle
        self.pose_roi = other.pose_roi
        self.img = other.img
        self.pose = other.pose
        self.active = other.active
        # self.start_time = other.start_time
        self.last_time = time()
        self.overlap = other.overlap

    def set_pose_roi(self, image: np.ndarray, roi_expansion: float) -> None:
        if self.pose_roi is not None:
            # print(f"Warning: pose rect already set for person {self.id} in camera {self.cam_id}.")
            return

        h, w = image.shape[:2]
        self.pose_roi = self.get_crop_rect(w, h, self.tracklet.roi, roi_expansion)

    def set_pose_image(self, image: np.ndarray) -> None:
        if self.img is not None:
            # print(f"Warning: pose image already set for person {self.id} in camera {self.cam_id}.")
            return

        if self.pose_roi is None:
            print(f"Warning: pose rect not set for person {self.id} in camera {self.cam_id}.")
            return

        self.img = self.get_cropped_image(image, self.pose_roi, 256)

    @staticmethod
    def create_cam_id(cam_id: int, tracklet_id: int) -> str:
        return f"{cam_id}_{tracklet_id}"

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

    def is_expired(self, threshold) -> bool:
        """Check if person hasn't been updated recently"""
        return time() - self.last_time > threshold

    @property
    def age(self) -> float:
        """Get how long this person has been tracked"""
        return time() - self.start_time

PersonCallback = Callable[[Person], None]
PersonDict = dict[int, Person]
PersonDictCallback = Callable[[PersonDict], None]


from threading import Lock

class PersonIdPool:
    def __init__(self, max_size: int) -> None:
        self._available = set(range(max_size))
        self._lock = Lock()

    def acquire(self) -> int:
        with self._lock:
            if not self._available:
                raise Exception("No more IDs available")
            min_id: int = min(self._available)
            self._available.remove(min_id)
            return min_id

    def release(self, obj: int) -> None:
        with self._lock:
            self._available.add(obj)

    def size(self) -> int:
        return len(self._available)