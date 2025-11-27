# from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, TypeVar, cast
from typing_extensions import get_args, get_origin

from enum import Enum
import json
import dataclasses

from modules.cam.depthcam.Definitions import FrameType, get_device_list
from modules.pose.detection.MMDetection import ModelType
from modules.tracker.TrackerBase import TrackerType

from modules.cam.Settings import Settings as CamSettings

T = TypeVar("T")

@dataclass
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


    # GENERAL
    num_players: int                   = None # type: ignore

    camera: CamSettings = CamSettings()

    art_type: 'Settings.ArtType'         = None # type: ignore

    #GUI
    gui_location_x: int                = None # type: ignore
    gui_location_y: int                = None # type: ignore
    gui_on_top: bool                   = None # type: ignore
    gui_default_file: str              = None # type: ignore

    # PATHS
    path_root: str                     = None # type: ignore
    path_model: str                    = None # type: ignore
    path_video: str                    = None # type: ignore
    path_temp: str                     = None # type: ignore
    path_file: str                     = None # type: ignore

    # TRACKING SETTINGS
    tracker_type: TrackerType          = None # type: ignore
    tracker_min_age: int               = None # type: ignore
    tracker_min_height: float          = None # type: ignore
    tracker_timeout: float             = None # type: ignore

    # POSE DETCTION SETTINGS
    pose_crop_expansion: float         = None # type: ignore
    pose_model_type: ModelType     = None # type: ignore
    pose_model_warmups: int             = None # type: ignore
    pose_active: bool                  = None # type: ignore
    pose_stream_capacity: int          = None # type: ignore
    pose_conf_threshold: float         = None # type: ignore
    pose_verbose: bool                 = None # type: ignore

    # POSE CORRELATION SETTINGS
    corr_rate_hz: float                = None # type: ignore
    corr_num_workers: int              = None # type: ignore
    corr_buffer_duration: int          = None # type: ignore
    corr_stream_timeout: float         = None # type: ignore
    corr_max_nan_ratio: float          = None # type: ignore
    corr_dtw_band: int                 = None # type: ignore
    corr_similarity_exp: float         = None # type: ignore
    corr_stream_capacity: int          = None # type: ignore

    # LIGHT SETTINGS
    light_resolution: int              = None # type: ignore
    light_rate: int                    = None # type: ignore

    # UDP SETTINGS
    udp_port: int                      = None # type: ignore
    udp_ips_light: str                 = None # type: ignore
    udp_ips_sound: str                 = None # type: ignore

    # RENDER SETTINGS
    render_title: str                  = None # type: ignore
    render_width: int                  = None # type: ignore
    render_height: int                 = None # type: ignore
    render_x: int                      = None # type: ignore
    render_y: int                      = None # type: ignore
    render_fullscreen: bool            = None # type: ignore
    render_fps: int                    = None # type: ignore
    render_v_sync: bool                = None # type: ignore
    render_cams_a_row: int             = None # type: ignore
    render_monitor: int                = None # type: ignore
    render_R_num: int                  = None # type: ignore
    render_secondary_list: list[int]   = None # type: ignore

    def check_values(self) -> None:
         for key, value in vars(self).items():
            if value is None:
                raise ValueError(f"'{key}' is not set")

    def check_cameras(self) -> None:
        available: list[str]  = get_device_list()
        selected: list[str] = []
        print(f"Available cameras: {available}")


    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(Settings.serialize(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'Settings':
        with open(path, "r") as f:
            data = json.load(f)
        return Settings.deserialize(data, Settings)

    @staticmethod
    def serialize(obj) -> Any:
        if isinstance(obj, Enum):
            return obj.name
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {k: Settings.serialize(v) for k, v in dataclasses.asdict(obj).items()}
        if isinstance(obj, dict):
            return {k: Settings.serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [Settings.serialize(v) for v in obj]
        return obj

    @staticmethod
    def deserialize(data: Any, target_type: type[T]) -> T:
        if dataclasses.is_dataclass(target_type):
            fields: tuple[dataclasses.Field[Any], ...] = dataclasses.fields(target_type)
            field_types: dict[str, Any] = {f.name: f.type for f in fields if f.init}
            kwargs: dict[str, Any] = {}
            for key, value in data.items():
                if key in field_types:
                    field_type: Any = field_types[key]
                    if isinstance(field_type, str):
                        kwargs[key] = value
                    else:
                        kwargs[key] = Settings.deserialize(value, field_type)
            # Create instance with only init fields
            instance = target_type(**kwargs)
            # Set non-init fields if present in data
            for f in fields:
                if not f.init and f.name in data:
                    setattr(instance, f.name, data[f.name])
            # Call __post_init__ if it exists
            post_init = getattr(instance, "__post_init__", None)
            if callable(post_init):
                post_init()
            return instance

        origin: Any = get_origin(target_type)
        if origin is list:
            args: tuple[Any, ...] = get_args(target_type)
            if args:
                item_type: Any = args[0]
                return cast(T, [Settings.deserialize(item, item_type) for item in data])

        if isinstance(target_type, type) and issubclass(target_type, Enum):
            return target_type[data]

        return cast(T, data)
