import numpy as np
from time import time

from typing import Callable
from modules.cam.depthcam.Definitions import Tracklet, Rect, Point3f
from modules.person.pose.PoseDefinitions import PoseList

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

class AnglePosition():
    def __init__(self, x_angle: float, y_pos: float, size: float) -> None:
        self.x_angle: float = x_angle   # normalised, 0.0 to 1.0
        self.y_pos: float = y_pos       # normalised, 0.0 to 1.0
        self.size: float = size         # max(height, width), normalised, 0.0 to 1.0

class Person():
    _id_counter = 0

    def __init__(self, id, cam_id: int, tracklet: Tracklet) -> None:
        self.id: int =                  id
        self.cam_id: int =              cam_id
        self.tracklet: Tracklet =       tracklet
        self.angle_pos: AnglePosition = AnglePosition(0, 0, 0)    # x: x_angle, y: , z: depth
        self.start_time: float =        time()
        self.last_time: float =         time()
        self.pose_rect: Rect | None =   None
        self.pose_image: np.ndarray | None = None
        self.pose: PoseList | None =    None
        self.active: bool =             True

    @staticmethod
    def create_cam_id(cam_id: int, tracklet_id: int) -> str:
        return f"{cam_id}_{tracklet_id}"

PersonCallback = Callable[[Person], None]
PersonDict = dict[int, Person]

class CamTracklet:
    def __init__(self, cam_id: int, tracklet: Tracklet) -> None:
        self.cam_id: int = cam_id
        self.tracklet: Tracklet = tracklet