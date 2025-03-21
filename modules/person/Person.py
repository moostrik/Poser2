import numpy as np
from time import time

from typing import Callable
from modules.cam.DepthAi.Definitions import Tracklet, Rect, Point3f
from modules.pose.PoseDefinitions import PoseList

class Person():
    _id_counter = 0

    def __init__(self, id, cam_id: int, tracklet: Tracklet) -> None:
        self.id: int =                  id
        self.cam_id: int =              cam_id
        self.tracklet: Tracklet =       tracklet
        self.angle_pos: Point3f =       Point3f(0, 0, 0)
        self.start_time: float =        time()
        self.last_time: float =         time()
        self.pose_rect: Rect | None =   None
        self.pose_image: np.ndarray | None = None
        self.pose: PoseList | None =    None

    @staticmethod
    def create_cam_id(cam_id: int, tracklet_id: int) -> str:
        return f"{cam_id}_{tracklet_id}"

PersonCallback = Callable[[Person], None]
PersonDict = dict[int, Person]

class CamTracklet:
    def __init__(self, cam_id: int, tracklet: Tracklet) -> None:
        self.cam_id: int = cam_id
        self.tracklet: Tracklet = tracklet