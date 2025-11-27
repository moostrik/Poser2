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
from modules.pose.Settings import Settings as PoseSettings
from modules.gui.PyReallySimpleGui import GuiSettings
from modules.inout.SoundOSC import SoundOSCConfig, DataType
from modules.render.Settings import Settings as RenderSettings

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


    art_type: 'Settings.ArtType'         = None # type: ignore

    # GENERAL
    num_players: int                   = None # type: ignore
    tracker_type: TrackerType          = None # type: ignore

    # PATHS
    path_root: str                     = None # type: ignore
    path_model: str                    = None # type: ignore
    path_video: str                    = None # type: ignore
    path_temp: str                     = None # type: ignore
    path_file: str                     = None # type: ignore

    # CAMERA SETTINGS
    camera: CamSettings = CamSettings()

    # POSE SETTINGS
    pose: PoseSettings = PoseSettings()

    # GUI SETTINGS
    gui: GuiSettings = GuiSettings()

    # INOUT SETTINGS
    sound_osc: SoundOSCConfig = SoundOSCConfig()

    # RENDER SETTINGS
    render: RenderSettings = RenderSettings()


    # POSE CORRELATION SETTINGS
    corr_rate_hz: float                = None # type: ignore
    corr_num_workers: int              = None # type: ignore
    corr_buffer_duration: int          = None # type: ignore
    corr_stream_timeout: float         = None # type: ignore
    corr_max_nan_ratio: float          = None # type: ignore
    corr_dtw_band: int                 = None # type: ignore
    corr_similarity_exp: float         = None # type: ignore
    corr_stream_capacity: int          = None # type: ignore

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
