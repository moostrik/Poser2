from dataclasses import dataclass, field
from typing import Any, TypeVar, cast
from typing_extensions import get_args, get_origin

from enum import Enum
import json
import dataclasses


from modules.tracker.TrackerBase import TrackerType

from modules.cam.Settings import Settings as CamSettings
from modules.pose.Settings import Settings as PoseSettings, ModelType
from modules.gui.PyReallySimpleGui import GuiSettings
from modules.inout.SoundOSC import SoundOSCConfig, DataType
from modules.render.Settings import Settings as RenderSettings
from modules.pose.pd_stream.PDStreamSettings import Settings as PDStreamSettings

T = TypeVar("T")

@dataclass
class Settings():
    # GENERAL
    num_players: int                   = 3
    tracker_type: TrackerType          = TrackerType.ONEPERCAM

    # CAMERA SETTINGS
    camera: CamSettings = CamSettings()

    # POSE SETTINGS
    pose: PoseSettings = PoseSettings()
    pd_stream: PDStreamSettings = PDStreamSettings()

    # GUI SETTINGS
    gui: GuiSettings = GuiSettings()

    # INOUT SETTINGS
    sound_osc: SoundOSCConfig = SoundOSCConfig()

    # RENDER SETTINGS
    render: RenderSettings = RenderSettings()


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
