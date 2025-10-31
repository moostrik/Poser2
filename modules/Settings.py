from __future__ import annotations

from enum import Enum

from modules.cam.depthcam.Definitions import FrameType, get_device_list
from modules.pose.PoseDetection import PoseModelType
from modules.tracker.TrackerBase import TrackerType

class Settings():
    class CoderType(Enum):
        CPU =   0
        GPU =   1
        iGPU =  2

    class CoderFormat(Enum):
        H264 = '.mp4'
        H265 = '.hevc'

    class ArtType(Enum):
        NONE = 0
        WS = 1
        HDT = 2

    def __init__(self) -> None:
        # GENERAL
        self.num_players: int                   = None # type: ignore
        self.art_type: Settings.ArtType         = None # type: ignore

        #GUI
        self.gui_location_x: int                = None # type: ignore
        self.gui_location_y: int                = None # type: ignore
        self.gui_on_top: bool                   = None # type: ignore
        self.gui_default_file: str              = None # type: ignore

        # PATHS
        self.path_root: str                     = None # type: ignore
        self.path_model: str                    = None # type: ignore
        self.path_video: str                    = None # type: ignore
        self.path_temp: str                     = None # type: ignore
        self.path_file: str                     = None # type: ignore

        # CAMERA SETTINGS
        self.camera_list: list[str]             = None # type: ignore
        self.camera_num: int                    = None # type: ignore
        self.camera_fps: float                  = None # type: ignore
        self.camera_square: bool                = None # type: ignore
        self.camera_color: bool                 = None # type: ignore
        self.camera_stereo: bool                = None # type: ignore
        self.camera_yolo: bool                  = None # type: ignore
        self.camera_show_stereo: bool           = None # type: ignore
        self.camera_simulation: bool            = None # type: ignore
        self.camera_passthrough: bool           = None # type: ignore
        self.camera_manual: bool                = None # type: ignore
        self.camera_flip_h: bool                = None # type: ignore
        self.camera_flip_v: bool                = None # type: ignore
        self.camera_perspective: float          = None # type: ignore

        # RECORDER AND PLAYER SETTINGS
        self.video_chunk_length: float          = None # type: ignore
        self.video_encoder: Settings.CoderType  = None # type: ignore
        self.video_decoder: Settings.CoderType  = None # type: ignore
        self.video_format: Settings.CoderFormat = None # type: ignore
        self.video_frame_types: list[FrameType] = None # type: ignore

        # TRACKING SETTINGS
        self.tracker_type: TrackerType          = None # type: ignore
        self.tracker_min_age: int               = None # type: ignore
        self.tracker_min_height: float          = None # type: ignore
        self.tracker_timeout: float             = None # type: ignore

        # POSE DETCTION SETTINGS
        self.pose_crop_expansion: float         = None # type: ignore
        self.pose_model_type: PoseModelType     = None # type: ignore
        self.pose_model_warmups: int             = None # type: ignore
        self.pose_active: bool                  = None # type: ignore
        self.pose_stream_capacity: int          = None # type: ignore
        self.pose_conf_threshold: float         = None # type: ignore
        self.pose_verbose: bool                 = None # type: ignore

        # POSE CORRELATION SETTINGS
        self.corr_rate_hz: float                = None # type: ignore
        self.corr_num_workers: int              = None # type: ignore
        self.corr_buffer_duration: int          = None # type: ignore
        self.corr_stream_timeout: float         = None # type: ignore
        self.corr_max_nan_ratio: float          = None # type: ignore
        self.corr_dtw_band: int                 = None # type: ignore
        self.corr_similarity_exp: float         = None # type: ignore
        self.corr_stream_capacity: int          = None # type: ignore

        # LIGHT SETTINGS
        self.light_resolution: int              = None # type: ignore
        self.light_rate: int                    = None # type: ignore

        # UDP SETTINGS
        self.udp_port: int                      = None # type: ignore
        self.udp_ips_light: list[str]           = None # type: ignore
        self.udp_ips_sound: list[str]           = None # type: ignore

        # RENDER SETTINGS
        self.render_title: str                  = None # type: ignore
        self.render_width: int                  = None # type: ignore
        self.render_height: int                 = None # type: ignore
        self.render_x: int                      = None # type: ignore
        self.render_y: int                      = None # type: ignore
        self.render_fullscreen: bool            = None # type: ignore
        self.render_fps: int                    = None # type: ignore
        self.render_v_sync: bool                = None # type: ignore
        self.render_cams_a_row: int             = None # type: ignore
        self.render_monitor: int                = None # type: ignore
        self.render_R_num: int                  = None # type: ignore
        self.render_secondary_list: list[int]   = None # type: ignore

    def check_values(self) -> None:
         for key, value in vars(self).items():
            if value is None:
                raise ValueError(f"'{key}' is not set")

    def check_cameras(self) -> None:
        available: list[str]  = get_device_list()
        selected: list[str] = []
        print(f"Available cameras: {available}")
        return
        for camera in self.camera_list:
            if camera not in available:
                print(f"Omitting camera '{camera}' not found in camera list")
            else:
                selected.append(camera)

        if not self.camera_passthrough:
            self.camera_list = selected
            self.camera_num = len(self.camera_list)
