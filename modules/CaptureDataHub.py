import numpy as np
from itertools import combinations
from typing import Optional, Tuple, Dict, List
from threading import Lock
from dataclasses import dataclass, field
from typing import Optional, TypeVar, Generic


from modules.cam.depthcam.Definitions import Tracklet as DepthTracklet, FrameType
from modules.tracker.Tracklet import Tracklet, TrackletDict
from modules.pose.Pose import Pose, PoseDict
from modules.pose.PoseStream import PoseStreamData
from modules.RenderDataHub import RenderDataHub
from modules.pose.correlation.PairCorrelationStream import PairCorrelationStreamData
from modules.pose.correlation.PairCorrelation import PairCorrelationBatch
from modules.WS.WSOutput import WSOutput
from modules.Settings import Settings

from modules.utils.HotReloadMethods import HotReloadMethods

T = TypeVar('T')

@dataclass
class DataItem(Generic[T]):
    value: T
    accessed: dict[str, bool] = field(default_factory=dict)

class CaptureDataHub:
    _consumer_counter: int = 0  # Class variable for unique IDs

    def __init__(self) -> None:
        self.mutex: Lock = Lock()

        # Data storage
        self.light_image: Dict[int, DataItem[WSOutput]] = {}
        self.cam_image: Dict[int, DataItem[np.ndarray]] = {}
        self.depth_tracklets: Dict[int, DataItem[List[DepthTracklet]]] = {}
        self.tracklets: Dict[int, DataItem[Tracklet]] = {}
        self.poses: Dict[int, DataItem[Pose]] = {}
        self.pose_streams: Dict[int, DataItem[PoseStreamData]] = {}
        self.pose_correlation: Dict[int, DataItem[PairCorrelationBatch]] = {}
        self.motion_correlation: Dict[int, DataItem[PairCorrelationBatch]] = {}

        self._hot_reload = HotReloadMethods(self.__class__, True, True)

    @classmethod
    def get_unique_consumer_key(cls) -> str:
        """Generate a unique consumer key using a counter."""
        cls._consumer_counter += 1
        return f"C_{cls._consumer_counter}"

    def _set_data_dict(self, data_dict: Dict[int, DataItem[T]], data_key: int, value: T) -> None:
        with self.mutex:
            data_dict[data_key] = DataItem(value)

    def _get_data_dict(self, data_dict: Dict[int, DataItem[T]], data_key: int, only_new_data: bool, consumer_key: str) -> Optional[T]:
        with self.mutex:
            item: Optional[DataItem[T]] = data_dict.get(data_key)
            if not item:
                return None
            if only_new_data and item.accessed.get(consumer_key, False):
                return None
            if only_new_data:
                item.accessed[consumer_key] = True
            return item.value

    # Audio-visual data management
    def set_light_image(self, value: WSOutput) -> None:
        self._set_data_dict(self.light_image, 0, value)

    def get_light_image(self, only_new_data: bool, consumer_key: str) -> Optional[WSOutput]:
        return self._get_data_dict(self.light_image, 0, only_new_data, consumer_key)

    # Camera image management
    def set_cam_image(self, key: int, frame_type: FrameType, value: np.ndarray) -> None:
        self._set_data_dict(self.cam_image, key, value)

    def get_cam_image(self, key: int, only_new_data: bool, consumer_key: str) -> Optional[np.ndarray]:
        return self._get_data_dict(self.cam_image, key, only_new_data, consumer_key)

    # Depth tracklet management
    def set_cam_tracklets(self, key: int, value: list[DepthTracklet]) -> None:
        self._set_data_dict(self.depth_tracklets, key, value)

    def get_depth_tracklets(self, key: int, only_new_data: bool, consumer_key: str) -> list[DepthTracklet]:
        result: List[DepthTracklet] | None = self._get_data_dict(self.depth_tracklets, key, only_new_data, consumer_key)
        return result if result is not None else []

    # Tracklet management
    def set_tracklets(self, tracklets: TrackletDict) -> None:
        for value in tracklets.values():
            self._set_data_dict(self.tracklets, value.id, value)

    def get_tracklet(self, id: int, only_new_data: bool, consumer_key: str) -> Optional[Tracklet]:
        return self._get_data_dict(self.tracklets, id, only_new_data, consumer_key)

    def get_tracklets(self) -> dict[int, Tracklet]:
        with self.mutex:
            return {k: v.value for k, v in self.tracklets.items() if v.value is not None}

    def get_tracklets_for_cam(self, cam_id: int) -> list[Tracklet]:
        with self.mutex:
            return [v.value for v in self.tracklets.values() if v.value is not None and v.value.cam_id == cam_id]

    def get_active_tracklets_for_cam(self, cam_id: int) -> list[Tracklet]:
        with self.mutex:
            return [v.value for v in self.tracklets.values() if v.value is not None and v.value.cam_id == cam_id and v.value.is_active]

    # Pose management
    def set_poses(self, poses: PoseDict) -> None:
        for pose in poses.values():
            self._set_data_dict(self.poses, pose.tracklet.id, pose)

    def get_pose(self, id: int, only_new_data: bool, consumer_key: str) -> Optional[Pose]:
        return self._get_data_dict(self.poses, id, only_new_data, consumer_key)

    def get_poses_for_cam(self, cam_id: int) -> List[Pose]:
        with self.mutex:
            return [v.value for v in self.poses.values() if v.value is not None and v.value.tracklet.cam_id == cam_id]

    # Pose window/stream management
    def set_pose_stream(self, value: PoseStreamData) -> None:
        self._set_data_dict(self.pose_streams, value.id, value)

    def get_pose_stream(self, id: int, only_new_data: bool, consumer_key: str) -> Optional[PoseStreamData]:
        return self._get_data_dict(self.pose_streams, id, only_new_data, consumer_key)

    # Correlation window management
    def set_pose_correlation(self, value: PairCorrelationBatch) -> None:
        self._set_data_dict(self.pose_correlation, 0, value)

    def get_pose_correlation(self, only_new_data: bool, consumer_key: str) -> Optional[PairCorrelationBatch]:
        return self._get_data_dict(self.pose_correlation, 0, only_new_data, consumer_key)

    def set_motion_correlation(self, value: PairCorrelationBatch) -> None:
        self._set_data_dict(self.motion_correlation, 0, value)

    def get_motion_correlation(self, only_new_data: bool, consumer_key: str) -> Optional[PairCorrelationBatch]:
        return self._get_data_dict(self.motion_correlation, 0, only_new_data, consumer_key)

