# Standard library imports
import logging
from enum import IntEnum, auto
from threading import Lock
from typing import Callable, Any

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

    pose_frame_R =      auto()   # sorted by track_id, has cam_id (RAW detection)
    pose_frame_C =      auto()   # sorted by track_id, has cam_id (CLEAN)
    pose_frame_S =      auto()   # sorted by track_id, has cam_id (SMOOTH)
    pose_frame_I =      auto()   # sorted by track_id, has cam_id (LERP)

    pose_window_R =     auto()   # sorted by track_id, {type[BaseFeature]: FeatureWindow} (RAW detection)
    pose_window_C =     auto()   # sorted by track_id, {type[BaseFeature]: FeatureWindow} (CLEAN)
    pose_window_S =     auto()   # sorted by track_id, {type[BaseFeature]: FeatureWindow} (SMOOTH)
    pose_window_I =     auto()   # sorted by track_id, {type[BaseFeature]: FeatureWindow} (LERP)

    sequence =          auto()   # SequencerState dataclass


# Stage → DataHubType lookup
_FRAME_TYPES: dict[Stage, DataHubType] = {
    Stage.RAW:      DataHubType.pose_frame_R,
    Stage.CLEAN:    DataHubType.pose_frame_C,
    Stage.SMOOTH:   DataHubType.pose_frame_S,
    Stage.LERP:     DataHubType.pose_frame_I,
}
_WINDOW_TYPES: dict[Stage, DataHubType] = {
    Stage.RAW:      DataHubType.pose_window_R,
    Stage.CLEAN:    DataHubType.pose_window_C,
    Stage.SMOOTH:   DataHubType.pose_window_S,
    Stage.LERP:     DataHubType.pose_window_I,
}

# DEPRECATED: Use PipelineStage instead
class PoseDataHubTypes(IntEnum):
    pose_R =      DataHubType.pose_frame_R.value
    pose_C =      DataHubType.pose_frame_C.value
    pose_S =      DataHubType.pose_frame_S.value
    pose_I =      DataHubType.pose_frame_I.value

# POSE_ENUMS: set[DataType] = {DataType.pose_R, DataType.pose_S, DataType.pose_I}
# SIMILARITY_ENUMS: set[DataType] = {DataType.sim_P, DataType.sim_M}

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

    def get_filtered(self, data_type: DataHubType, filter_fn: Callable[[Any], bool]) -> set[Any]:
        with self.mutex:
            return {v for v in self._data.get(data_type, {}).values() if filter_fn(v)}

    def has_item(self, data_type: DataHubType, key: int = 0) -> bool:
        with self.mutex:
            return key in self._data.get(data_type, {})

    # CONVENIENCE GETTERS
    def get_items_for_cam(self, data_type: DataHubType, cam_id: int) -> set[Any]:
        """ this works on tracklets and poses """
        return self.get_filtered(data_type, lambda v: hasattr(v, "cam_id") and v.cam_id == cam_id)

    def has_items_for_cam(self, data_type: DataHubType, cam_id: int) -> bool:
        """ this works on tracklets and poses """
        return any(self.get_filtered(data_type, lambda v: hasattr(v, "cam_id") and v.cam_id == cam_id))

    def get_poses_for_cam(self, stage: Stage, cam_id: int) -> set[Any]:
        """Get all pose frames for a specific stage that belong to a camera."""
        return self.get_items_for_cam(_FRAME_TYPES[stage], cam_id)

    # POSE GETTERS
    def get_poses(self, stage: Stage) -> dict[int, Any]:
        """Get all pose frames for a specific stage."""
        return self.get_dict(_FRAME_TYPES[stage])

    def get_pose(self, stage: Stage, track_id: int) -> Any | None:
        """Get a single pose frame for a specific stage and track."""
        return self.get_item(_FRAME_TYPES[stage], track_id)

    def get_pose_count(self, stage: Stage) -> int:
        """Get the number of active poses for a specific stage."""
        with self.mutex:
            return len(self._data.get(_FRAME_TYPES[stage], {}))

    # FEATURE WINDOW GETTERS (stored as _data[pose_window_X] = {track_id: {type[BaseFeature]: FeatureWindow}})
    def get_feature_window(self, stage: Stage, feature_type: type[BaseFeature], track_id: int) -> Any | None:
        """Get a single feature window for a specific stage, feature type, and track."""
        with self.mutex:
            track_windows = self._data.get(_WINDOW_TYPES[stage], {}).get(track_id)
            if track_windows is None:
                return None
            return track_windows.get(feature_type)

    def get_feature_windows_for_track(self, stage: Stage, track_id: int) -> dict[type[BaseFeature], FeatureWindow]:
        """Get all feature windows for a specific stage and track."""
        with self.mutex:
            return dict(self._data.get(_WINDOW_TYPES[stage], {}).get(track_id, {}))

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

