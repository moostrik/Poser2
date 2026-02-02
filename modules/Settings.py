import dataclasses
from dataclasses import dataclass, field
from enum import Enum
import json
import os
from pathlib import Path
from typing import Any, TypeVar, cast
from typing_extensions import get_args, get_origin


from modules.cam.Config import Config as CamConfig
from modules.gui.PyReallySimpleGui import GuiConfig
from modules.inout import OscSoundConfig
from modules.inout.ArtNetLed import ArtNetLedConfig

from modules.render.Settings import Settings as RenderSettings
from modules.pose.Settings import Settings as PoseSettings, ModelType
from modules.tracker.TrackerBase import TrackerType

T = TypeVar("T")

@dataclass
class Settings():
    # GENERAL
    num_players: int =          field(default=3)
    tracker_type: TrackerType = field(default=TrackerType.ONEPERCAM)

    # CAMERA SETTINGS
    camera: CamConfig = CamConfig()

    # POSE SETTINGS
    pose: PoseSettings = PoseSettings()

    # GUI SETTINGS
    gui: GuiConfig = GuiConfig()

    # INOUT SETTINGS
    sound_osc: OscSoundConfig = OscSoundConfig()
    artnet_leds: list[ArtNetLedConfig] = field(default_factory=list)

    # RENDER SETTINGS
    render: RenderSettings = RenderSettings()


    def save(self, path: str, sort_keys: bool = False) -> None:
        # Serialize with sorted keys
        json_str = json.dumps(Settings.serialize(self), indent=2, sort_keys=sort_keys)
        # Add a blank line between each root element
        lines = json_str.splitlines()
        new_lines = []
        for i, line in enumerate(lines):
            new_lines.append(line)
            # Add a blank line after each top-level key (except the last and braces)
            if (
                line.endswith(',') and
                lines[i + 1].startswith('  "')  # next line is another root key
            ):
                new_lines.append('')
        with open(path, "w") as f:
            f.write('\n'.join(new_lines))

    @staticmethod
    def load(path: str) -> "Settings":
        with open(path, "r") as f:
            data: Any = json.load(f)
        return Settings.deserialize(data, Settings, validate=True)

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
    def deserialize(data: Any, target_type: type[T], validate: bool = False, path: str = "") -> T:
        if dataclasses.is_dataclass(target_type):
            fields: tuple[dataclasses.Field[Any], ...] = dataclasses.fields(target_type)
            field_types: dict[str, Any] = {f.name: f.type for f in fields if f.init}

            if validate:
                # Check for extra keys in data
                extra_keys = set(data.keys()) - set(field_types.keys())
                if extra_keys:
                    location = f" at '{path}'" if path else ""
                    print(f"SETTINGS WARNING: Extra keys in config{location}: {', '.join(sorted(extra_keys))}")

                # Check for missing keys in data (even those with defaults)
                missing_keys = set(field_types.keys()) - set(data.keys())
                if missing_keys:
                    location = f" at '{path}'" if path else ""
                    print(f"SETTINGS WARNING: Missing keys in config{location}: {', '.join(sorted(missing_keys))}")

            kwargs: dict[str, Any] = {}
            for key, value in data.items():
                if key in field_types:
                    field_type: Any = field_types[key]
                    if isinstance(field_type, str):
                        kwargs[key] = value
                    else:
                        new_path = f"{path}.{key}" if path else key
                        kwargs[key] = Settings.deserialize(value, field_type, validate, new_path)
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
                return cast(T, [Settings.deserialize(item, item_type, validate, path) for item in data])

        if isinstance(target_type, type) and issubclass(target_type, Enum):
            return target_type[data]

        return cast(T, data)

    @staticmethod
    def make_paths_relative(obj, path_root: Path | str) -> None:
        path_root = Path(path_root)
        if not path_root.exists():
            raise FileNotFoundError(f"path_root does not exist: {path_root}")
        if dataclasses.is_dataclass(obj):
            for field_ in dataclasses.fields(obj):
                value = getattr(obj, field_.name)
                if isinstance(value, str) and field_.name.endswith('_path'):
                    abs_path = Path(value)
                    try:
                        rel_path = abs_path.relative_to(path_root)
                        setattr(obj, field_.name, str(rel_path))
                    except ValueError:
                        pass
                elif dataclasses.is_dataclass(value):
                    Settings.make_paths_relative(value, path_root)
                elif isinstance(value, list):
                    for item in value:
                        Settings.make_paths_relative(item, path_root)
                elif isinstance(value, dict):
                    for item in value.values():
                        Settings.make_paths_relative(item, path_root)

    @staticmethod
    def make_paths_absolute(obj, path_root: Path | str) -> None:
        path_root = Path(path_root)
        if not path_root.exists():
            raise FileNotFoundError(f"path_root does not exist: {path_root}")
        if dataclasses.is_dataclass(obj):
            for field_ in dataclasses.fields(obj):
                value = getattr(obj, field_.name)
                if isinstance(value, str) and field_.name.endswith('_path'):
                    path = Path(value)
                    if not path.is_absolute():
                        abs_path = path_root / path
                        setattr(obj, field_.name, str(abs_path))
                elif dataclasses.is_dataclass(value):
                    Settings.make_paths_absolute(value, path_root)
                elif isinstance(value, list):
                    for item in value:
                        Settings.make_paths_absolute(item, path_root)
                elif isinstance(value, dict):
                    for item in value.values():
                        Settings.make_paths_absolute(item, path_root)
