from __future__ import annotations
from enum import Enum

from modules.cam.depthcam.Definitions import get_device_list
from modules.person.pose.PoseDetection import ModelType
from modules.cam.depthcam.Definitions import FrameType

class Settings():
    class CoderType(Enum):
        CPU =   0
        GPU =   1
        iGPU =  2

    def __init__(self) -> None:
        # PATHS
        self.root_path: str
        self.model_path: str
        self.video_path: str
        self.temp_path: str
        self.file_path: str

        # CAMERA SETTINGS
        self.camera_list: list[str]
        self.num_cams: int
        self.fps: int
        self.color: bool
        self.stereo: bool
        self.person: bool
        self.lowres: bool
        self.show_stereo: bool
        self.simulation: bool
        self.passthrough: bool

        # DETECTION SETTINGS
        self.num_players: int
        self.model_type: ModelType
        self.pose: bool
        self.num_cams: int

        # RECORDER AND PLAYER SETTINGS
        self.chunk_length: float
        self.encoder: Settings.CoderType
        self.decoder: Settings.CoderType
        self.frame_types: list[FrameType]

    def check_values(self) -> None:
        for key, value in vars(self).items():
            if value is None:
                raise ValueError(f"'{key}' is not set")

    def check_cameras(self) -> None:
        available: list[str]  = get_device_list()
        selected: list[str] = []
        for camera in self.camera_list:
            if camera not in available:
                print(f"Omitting camera '{camera}' not found in camera list")
            else:
                selected.append(camera)

        if not self.passthrough:
            self.camera_list = selected
            self.num_cams = len(self.camera_list)
