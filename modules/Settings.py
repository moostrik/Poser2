from __future__ import annotations

from enum import Enum
from typing import Optional

from modules.cam.depthcam.Definitions import get_device_list
from modules.person.pose.Detection import ModelType
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
        self.path_root: Optional[str]                       = None
        self.path_model: Optional[str]                      = None
        self.path_video: Optional[str]                      = None
        self.path_temp: Optional[str]                       = None
        self.path_file: Optional[str]                       = None

        # CAMERA SETTINGS
        self.camera_list: Optional[list[str]]               = None
        self.camera_num: Optional[int]                      = None
        self.camera_fps: Optional[int]                      = None
        self.camera_square: Optional[bool]                  = None
        self.camera_color: Optional[bool]                   = None
        self.camera_stereo: Optional[bool]                  = None
        self.camera_yolo: Optional[bool]                    = None
        self.camera_show_stereo: Optional[bool]             = None
        self.camera_simulation: Optional[bool]              = None
        self.camera_passthrough: Optional[bool]             = None
        self.camera_manual: Optional[bool]                  = None

        # DETECTION SETTINGS
        self.pose_num: Optional[int]                        = None
        self.pose_model_type: Optional[ModelType]           = None
        self.pose_active: Optional[bool]                    = None

        # RECORDER AND PLAYER SETTINGS
        self.video_chunk_length: Optional[float]            = None
        self.video_encoder: Optional[Settings.CoderType]    = None
        self.video_decoder: Optional[Settings.CoderType]    = None
        self.video_format: Optional[Settings.CoderFormat]   = None
        self.video_frame_types: Optional[list[FrameType]]   = None

        # RENDER SETTINGS
        self.render_title: Optional[str]                    = None
        self.render_width: Optional[int]                    = None
        self.render_height: Optional[int]                   = None
        self.render_x: Optional[int]                        = None
        self.render_y: Optional[int]                        = None
        self.render_fullscreen: Optional[bool]              = None
        self.render_fps: Optional[int]                      = None
        self.render_v_sync: Optional[bool]                  = None
        self.render_cams_a_row: Optional[int]               = None

        # LIGHT SETTINGS
        self.light_resolution: Optional[int]                = None
        self.light_rate: Optional[int]                      = None


    def check_values(self) -> None:
         for key, value in vars(self).items():
            if value is None:
                raise ValueError(f"'{key}' is not set")

    def check_cameras(self) -> None:
        if not self.camera_list:
            return
        available: list[str]  = get_device_list()
        selected: list[str] = []
        for camera in self.camera_list:
            if camera not in available:
                print(f"Omitting camera '{camera}' not found in camera list")
            else:
                selected.append(camera)

        if not self.camera_passthrough:
            self.camera_list = selected
            self.camera_num = len(self.camera_list)
