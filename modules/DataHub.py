# Standard library imports
from enum import IntEnum
from threading import Lock
from traceback import print_exc
from typing import Callable, Any

# Third party imports
import numpy as np

# Local application imports for setter types
from modules.cam.depthcam.Definitions import Tracklet as DepthTracklet
from modules.pose.Pose import PoseDict
from modules.pose.pd_stream.PDStream import PDStreamData
from modules.pose.similarity import SimilarityBatch
from modules.tracker.Tracklet import TrackletDict
from modules.WS.WSOutput import WSOutput


class DataType(IntEnum):
    light_image = 0         # single image
    cam_image = 1           # sorted by cam_id
    depth_tracklet = 2      # sorted by cam_id
    tracklet = 3            # sorted by track_id, has cam_id (currelntly named id -> RENAME)
    R_pose = 4              # sorted by track_id, has cam_id
    S_pose = 5              # sorted by track_id, has cam_id
    I_pose = 6              # sorted by track_id, has cam_id
    pose_correlation = 7    # single SimilarityBatch
    motion_correlation = 8  # single SimilarityBatch
    pose_stream = 9         # sorted by track_id


class DataHub:
    def __init__(self) -> None:
        self.mutex: Lock = Lock()
        self._data: dict[DataType, dict[int, Any]] = {}

        self._update_callback_lock = Lock()
        self._update_callbacks: set[Callable] = set()

    # GENERIC GETTERS
    def get_item(self, data_type: DataType, key: int = 0) -> Any | None:
        with self.mutex:
            return self._data.get(data_type, {}).get(key)

    def get_dict(self, data_type: DataType) -> dict[int, Any]:
        with self.mutex:
            return dict(self._data.get(data_type, {}))

    def get_filtered(self, data_type: DataType, filter_fn: Callable[[Any], bool]) -> list[Any]:
        with self.mutex:
            return [v for v in self._data.get(data_type, {}).values() if filter_fn(v)]

    def has_item(self, data_type: DataType, key: int = 0) -> bool:
        with self.mutex:
            return key in self._data.get(data_type, {})

    # CONVENIENCE GETTERS
    # def get_items_for_cam(self, data_type: DataType, cam_id: int) -> list[Any]:
    #     """ this works on tracklets and poses """
    #     return self.get_filtered(data_type, lambda v: hasattr(v, "cam_id") and v.cam_id == cam_id)

    # def has_items_for_cam(self, data_type: DataType, cam_id: int) -> bool:
    #     """ this works on tracklets and poses """
    #     return any(self.get_filtered(data_type, lambda v: hasattr(v, "cam_id") and v.cam_id == cam_id))

    # GENERIC SETTERS
    def set_item(self, data_type: DataType, key: int, value: Any) -> None:
        """Set a single item without replacing the entire dict."""
        with self.mutex:
            if data_type not in self._data:
                self._data[data_type] = {}
            self._data[data_type][key] = value

    def set_dict(self, data_type: DataType, values: dict[int, Any]) -> None:
        """Replace entire dict for a data type."""
        with self.mutex:
            self._data[data_type] = dict(values)

    # TYPE-SPECIFIC SETTERS
    def set_light_image(self, value: WSOutput) -> None:
        self.set_dict(DataType.light_image, {0: value})

    def set_cam_image(self, key: int, frame_type, value: np.ndarray) -> None:
        self.set_item(DataType.cam_image, key, value)

    def set_depth_tracklets(self, key: int, value: list[DepthTracklet]) -> None:
        self.set_item(DataType.depth_tracklet, key, value)

    def set_tracklets(self, tracklets: TrackletDict) -> None:
        self.set_dict(DataType.tracklet, tracklets)

    def set_poses(self, data_type: DataType, poses: PoseDict) -> None:
        self.set_dict(data_type, poses)

    def set_pose_correlation(self, value: SimilarityBatch) -> None:
        self.set_dict(DataType.pose_correlation, {0: value})

    def set_motion_correlation(self, value: SimilarityBatch) -> None:
        self.set_dict(DataType.motion_correlation, {0: value})

    def set_pose_stream(self, pd_stream: PDStreamData) -> None:
        self.set_item(DataType.pose_stream, pd_stream.track_id, pd_stream)

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