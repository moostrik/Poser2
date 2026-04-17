# Standard library imports
import logging
from enum import IntEnum, auto
from threading import Lock
from typing import Any

# Local application imports
from modules.pose.frame import FeatureWindow, FrameWindowDict
from modules.pose.features.base import BaseFeature


class Stage(IntEnum):
    RAW =       0
    CLEAN =     auto()
    SMOOTH =    auto()
    LERP =      auto()


class DataHubType(IntEnum):
    cam_image =         auto()   # sorted by cam_id, raw camera images
    depth_tracklet =    auto()   # sorted by cam_id

    tracklet =          auto()   # sorted by track_id, has cam_id
    gpu_frames =        auto()   # sorted by track_id, GPU frames with crops

    frame_raw =         auto()   # sorted by track_id, has cam_id (RAW detection)
    frame_clear =       auto()   # sorted by track_id, has cam_id (CLEAN)
    frame_smooth =      auto()   # sorted by track_id, has cam_id (SMOOTH)
    frame_lerp =        auto()   # sorted by track_id, has cam_id (LERP)

    window_raw =        auto()   # sorted by track_id, {type[BaseFeature]: FeatureWindow} (RAW detection)
    window_clear =      auto()   # sorted by track_id, {type[BaseFeature]: FeatureWindow} (CLEAN)
    window_smooth =     auto()   # sorted by track_id, {type[BaseFeature]: FeatureWindow} (SMOOTH)
    window_lerp =       auto()   # sorted by track_id, {type[BaseFeature]: FeatureWindow} (LERP)

    sequence =          auto()   # SequencerState dataclass


# Stage → DataHubType lookup
FRAME_TYPES: dict[Stage, DataHubType] = {
    Stage.RAW:      DataHubType.frame_raw,
    Stage.CLEAN:    DataHubType.frame_clear,
    Stage.SMOOTH:   DataHubType.frame_smooth,
    Stage.LERP:     DataHubType.frame_lerp,
}
_WINDOW_TYPES: dict[Stage, DataHubType] = {
    Stage.RAW:      DataHubType.window_raw,
    Stage.CLEAN:    DataHubType.window_clear,
    Stage.SMOOTH:   DataHubType.window_smooth,
    Stage.LERP:     DataHubType.window_lerp,
}


class DataHub:
    _logger = logging.getLogger(__name__)

    def __init__(self) -> None:
        self.mutex: Lock = Lock()
        self._data: dict[DataHubType, dict[int, Any]] = {}

    # GENERIC GETTERS
    def get_item(self, data_type: DataHubType, key: int = 0) -> Any | None:
        with self.mutex:
            return self._data.get(data_type, {}).get(key)

    def get_dict(self, data_type: DataHubType) -> dict[int, Any]:
        with self.mutex:
            return dict(self._data.get(data_type, {}))

    # POSE GETTERS
    def get_pose(self, stage: Stage, track_id: int) -> Any | None:
        """Get a single pose frame for a specific stage and track."""
        return self.get_item(FRAME_TYPES[stage], track_id)

    # FEATURE WINDOW GETTERS (stored as _data[pose_window_X] = {track_id: {type[BaseFeature]: FeatureWindow}})
    def get_feature_window(self, stage: Stage, feature_type: type[BaseFeature], track_id: int) -> Any | None:
        """Get a single feature window for a specific stage, feature type, and track."""
        with self.mutex:
            track_windows = self._data.get(_WINDOW_TYPES[stage], {}).get(track_id)
            if track_windows is None:
                return None
            return track_windows.get(feature_type)

    # GENERIC SETTERS
    def set_item(self, data_type: DataHubType, key: int, value: object) -> None:
        """Set a single item without replacing the entire dict."""
        with self.mutex:
            if data_type not in self._data:
                self._data[data_type] = {}
            self._data[data_type][key] = value

    def set_dict(self, data_type: DataHubType, values: dict[int, Any]) -> None:
        """Replace entire dict for a data type."""
        with self.mutex:
            self._data[data_type] = dict(values)

    # POSE SETTERS
    def set_pose_windows(self, stage: Stage, windows: FrameWindowDict) -> None:
        """Store feature windows. Pivots {type[BaseFeature]: {track_id: FeatureWindow}} to track-first for storage."""
        pivoted: dict[int, Any] = {}
        for field, track_windows in windows.items():
            for track_id, window in track_windows.items():
                if track_id not in pivoted:
                    pivoted[track_id] = {}
                pivoted[track_id][field] = window
        self.set_dict(_WINDOW_TYPES[stage], pivoted)

