import numpy as np
from time import sleep

from typing import Callable
from modules.cam.DepthAi.Definitions import Tracklet, Rect

from modules.pose.PoseDefinitions import PoseList

class Message():
    _id_counter = 0

    def __init__(self, cam_id: int, tracklet: Tracklet) -> None:
        self.id: str =                  self.create_unique_id(cam_id, tracklet.id)
        self.cam_id: int =              cam_id
        self.tracklet: Tracklet =       tracklet
        self.image: np.ndarray | None = None
        self.pose: PoseList | None =    None

    @staticmethod
    def create_unique_id(cam_id: int, tracklet_id: int) -> str:
        return f"{cam_id}_{tracklet_id}"

MessageCallback = Callable[[Message], None]