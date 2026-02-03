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
from modules.pose.Frame import FrameDict
from modules.tracker.Tracklet import TrackletDict
from modules.utils.Timer import TimerState


class DataHubType(IntEnum):
    cam_image =             auto()   # sorted by cam_id, raw camera images
    depth_tracklet =        auto()   # sorted by cam_id
    tracklet =              auto()   # sorted by track_id, has cam_id
    pose_R =                auto()   # sorted by track_id, has cam_id
    pose_S =                auto()   # sorted by track_id, has cam_id
    pose_I =                auto()   # sorted by track_id, has cam_id

    gpu_frames =            auto()   # sorted by track_id, GPU frames with crops
    mask_tensor =           auto()   # sorted by track_id, GPU tensors (H, W) FP16
    flow_tensor =           auto()   # sorted by track_id, GPU tensors (H, W, 2) FP16
    flow_images =           auto()   # sorted by track_id, flow visualization images

    angle_window =          auto()   # sorted by track_id, FeatureWindow[AngleLandmark]
    angle_vel_window =      auto()   # sorted by track_id, FeatureWindow[AngleLandmark]
    angle_motion_window =   auto()   # sorted by track_id, FeatureWindow[AngleLandmark]
    bbox_window =           auto()   # sorted by track_id, FeatureWindow[BBoxElement]
    similarity_window =     auto()   # sorted by track_id, FeatureWindow[AngleLandmark]

    timer_state =           auto()   # TimerState int value (IDLE=0, RUNNING=1, INTERMEZZO=2)
    timer_time =            auto()   # float, elapsed time in seconds

class PoseDataHubTypes(IntEnum):
    pose_R =      DataHubType.pose_R.value
    pose_S =      DataHubType.pose_S.value
    pose_I =      DataHubType.pose_I.value

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

    # WINDOW CONVENIENCE GETTERS
    def get_angle_windows(self) -> dict[int, Any]:
        """Get all angle feature windows."""
        return self.get_dict(DataHubType.angle_window)

    def get_angle_vel_windows(self) -> dict[int, Any]:
        """Get all angle velocity feature windows."""
        return self.get_dict(DataHubType.angle_vel_window)

    def get_angle_window(self, track_id: int) -> Any | None:
        """Get angle window for specific track."""
        return self.get_item(DataHubType.angle_window, track_id)

    def get_angle_vel_window(self, track_id: int) -> Any | None:
        """Get angle velocity window for specific track."""
        return self.get_item(DataHubType.angle_vel_window, track_id)

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

    def set_poses(self, data_type: DataHubType, poses: FrameDict) -> None:
        self.set_dict(data_type, poses)

    def set_mask_tensors(self, masks: dict[int, Tensor]) -> None:
        self.set_dict(DataHubType.mask_tensor, masks)

    def set_flow_tensors(self, flows: dict[int, Tensor]) -> None:
        self.set_dict(DataHubType.flow_tensor, flows)

    def set_flow_images(self, _: FrameDict, images: dict[int, tuple[np.ndarray, np.ndarray]]) -> None:
        self.set_dict(DataHubType.flow_images, images)

    def set_gpu_frames(self, _: FrameDict, gpu_frames) -> None:
        """Store GPU frame data. Expects GPUFrameDict from GPUCropProcessor."""
        self.set_dict(DataHubType.gpu_frames, gpu_frames)

    # WINDOWS
    def set_angle_windows(self, windows) -> None:
        """Store angle feature windows. Expects dict[int, FeatureWindow]."""
        self.set_dict(DataHubType.angle_window, windows)

    def set_angle_vel_windows(self, windows) -> None:
        """Store angle velocity feature windows. Expects dict[int, FeatureWindow]."""
        self.set_dict(DataHubType.angle_vel_window, windows)

    def set_angle_motion_windows(self, windows) -> None:
        """Store angle motion feature windows. Expects dict[int, FeatureWindow]."""
        self.set_dict(DataHubType.angle_motion_window, windows)

    def set_bbox_windows(self, windows) -> None:
        """Store bounding box feature windows. Expects dict[int, FeatureWindow]."""
        self.set_dict(DataHubType.bbox_window, windows)

    def set_similarity_windows(self, windows) -> None:
        """Store angle symmetry feature windows. Expects dict[int, FeatureWindow]."""
        self.set_dict(DataHubType.similarity_window, windows)

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