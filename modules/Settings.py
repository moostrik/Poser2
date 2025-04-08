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

    class CoderFormat(Enum):
        H264 = '.mp4'
        H265 = '.hevc'

    def __init__(self) -> None:
        # PATHS
        self.root_path: str                 = None # type: ignore
        self.model_path: str                = None # type: ignore
        self.video_path: str                = None # type: ignore
        self.temp_path: str                 = None # type: ignore
        self.file_path: str                 = None # type: ignore

        # CAMERA SETTINGS
        self.camera_list: list[str]         = None # type: ignore
        self.num_cams: int                  = None # type: ignore
        self.fps: int                       = None # type: ignore
        self.color: bool                    = None # type: ignore
        self.stereo: bool                   = None # type: ignore
        self.person: bool                   = None # type: ignore
        self.lowres: bool                   = None # type: ignore
        self.show_stereo: bool              = None # type: ignore
        self.simulation: bool               = None # type: ignore
        self.passthrough: bool              = None # type: ignore

        # DETECTION SETTINGS
        self.num_players: int               = None # type: ignore
        self.model_type: ModelType          = None # type: ignore
        self.pose: bool                     = None # type: ignore

        # RECORDER AND PLAYER SETTINGS
        self.chunk_length: float            = None # type: ignore
        self.encoder: Settings.CoderType    = None # type: ignore
        self.decoder: Settings.CoderType    = None # type: ignore
        self.format: Settings.CoderFormat   = None # type: ignore
        self.frame_types: list[FrameType]   = None # type: ignore


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
