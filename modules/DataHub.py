# Standard library imports
from enum import IntEnum, auto
from threading import Lock
from traceback import print_exc
from typing import Callable, Any

# Third party imports
import numpy as np
from torch import Tensor

# Local application imports for setter types
from modules.cam.depthcam.Definitions import Tracklet as DepthTracklet
from modules.pose.Frame import FrameDict, FrameField
from modules.tracker.Tracklet import TrackletDict
from modules.utils.Timer import TimerState


class Stage(IntEnum):
    RAW =       0
    SMOOTH =    auto()
    LERP =      auto()


class DataHubType(IntEnum):
    cam_image =         auto()   # sorted by cam_id, raw camera images
    depth_tracklet =    auto()   # sorted by cam_id
    tracklet =          auto()   # sorted by track_id, has cam_id

    gpu_frames =        auto()   # sorted by track_id, GPU frames with crops
    flow_tensor =       auto()   # sorted by track_id, GPU tensors (H, W, 2) FP16

    pose_frame_R =      auto()   # sorted by track_id, has cam_id
    pose_frame_S =      auto()   # sorted by track_id, has cam_id
    pose_frame_I =      auto()   # sorted by track_id, has cam_id

    pose_window_R =     auto()   # sorted by track_id, {FrameField: FeatureWindow}
    pose_window_S =     auto()   # sorted by track_id, {FrameField: FeatureWindow}
    pose_window_I =     auto()   # sorted by track_id, {FrameField: FeatureWindow}

    timer_state =       auto()   # TimerState int value (IDLE=0, RUNNING=1, INTERMEZZO=2)
    timer_time =        auto()   # float, elapsed time in seconds


# Stage → DataHubType lookup
_FRAME_TYPES: dict[Stage, DataHubType] = {
    Stage.RAW:          DataHubType.pose_frame_R,
    Stage.SMOOTH:       DataHubType.pose_frame_S,
    Stage.LERP: DataHubType.pose_frame_I,
}
_WINDOW_TYPES: dict[Stage, DataHubType] = {
    Stage.RAW:          DataHubType.pose_window_R,
    Stage.SMOOTH:       DataHubType.pose_window_S,
    Stage.LERP: DataHubType.pose_window_I,
}

# DEPRECATED: Use PipelineStage instead
class PoseDataHubTypes(IntEnum):
    pose_R =      DataHubType.pose_frame_R.value
    pose_S =      DataHubType.pose_frame_S.value
    pose_I =      DataHubType.pose_frame_I.value

# POSE_ENUMS: set[DataType] = {DataType.pose_R, DataType.pose_S, DataType.pose_I}
# SIMILARITY_ENUMS: set[DataType] = {DataType.sim_P, DataType.sim_M}

class DataHub:
    def __init__(self) -> None:
        self.mutex: Lock = Lock()
        self._data: dict[DataHubType, dict[int, Any]] = {}

        self._update_callback_lock = Lock()
        self._update_callbacks: set[Callable] = set()

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

    # FEATURE WINDOW GETTERS (stored as _data[pose_window_X] = {track_id: {FrameField: FeatureWindow}})
    def get_feature_window(self, stage: Stage, field: FrameField, track_id: int) -> Any | None:
        """Get a single feature window for a specific stage, field, and track."""
        with self.mutex:
            track_windows = self._data.get(_WINDOW_TYPES[stage], {}).get(track_id)
            if track_windows is None:
                return None
            return track_windows.get(field)

    def get_feature_windows_for_track(self, stage: Stage, track_id: int) -> dict[FrameField, Any]:
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

    # TYPE-SPECIFIC SETTERS WITH CAM_ID KEY
    def set_cam_image(self, key: int, frame_type, value: np.ndarray) -> None:
        self.set_item(DataHubType.cam_image, key, value)

    def set_depth_tracklets(self, key: int, value: list[DepthTracklet]) -> None:
        self.set_item(DataHubType.depth_tracklet, key, value)

    # TYPE-SPECIFIC SETTERS WITH TRACK_ID KEY
    def set_tracklets(self, tracklets: TrackletDict) -> None:
        self.set_dict(DataHubType.tracklet, tracklets)

    def set_flow_tensors(self, flows: dict[int, Tensor]) -> None:
        self.set_dict(DataHubType.flow_tensor, flows)

    def set_gpu_frames(self, _: FrameDict, gpu_frames) -> None:
        """Store GPU frame data. Expects GPUFrameDict from GPUCropProcessor."""
        self.set_dict(DataHubType.gpu_frames, gpu_frames)

    # POSE SETTERS
    def set_poses(self, stage: Stage, poses: FrameDict) -> None:
        self.set_dict(_FRAME_TYPES[stage], poses)

    def set_feature_windows(self, stage: Stage, windows: dict[int, dict[FrameField, Any]]) -> None:
        """Store feature windows. Expects {track_id: {FrameField: FeatureWindow}}."""
        self.set_dict(_WINDOW_TYPES[stage], windows)

    # TIMER
    def set_timer_state(self, state: TimerState) -> None:
        """Store timer state as int value."""
        self.set_item(DataHubType.timer_state, 0, state.value)

    def set_timer_time(self, time: float) -> None:
        """Store timer elapsed time in seconds."""
        self.set_item(DataHubType.timer_time, 0, time)

    # UPDATE CALLBACK
    def notify_update(self) -> None:
        with self._update_callback_lock:
            for callback in self._update_callbacks:
                try:
                    callback()
                except Exception as e:
                    print(f"{self.__class__.__name__}: Error in callback: {e}")
                    print_exc()

    def add_update_callback(self, callback: Callable) -> None:
        """Register output callback."""
        with self._update_callback_lock:
            self._update_callbacks.add(callback)

    def remove_update_callback(self, callback: Callable) -> None:
        """Unregister output callback."""
        with self._update_callback_lock:
            self._update_callbacks.discard(callback)