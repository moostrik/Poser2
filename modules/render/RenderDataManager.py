import numpy as np
from itertools import combinations
from typing import Optional, Tuple, Dict, List
from threading import Lock
from dataclasses import dataclass
from typing import Optional, TypeVar, Generic


from modules.cam.depthcam.Definitions import Tracklet as CamTracklet, FrameType
from modules.tracker.Tracklet import Tracklet
from modules.pose.PoseDefinitions import Pose, PosePoints, PoseEdgeIndices
from modules.pose.PoseStream import PoseStreamData
from modules.correlation.PairCorrelationStream import PairCorrelationStreamData
from modules.av.Definitions import AvOutput
from modules.Settings import Settings

T = TypeVar('T')

@dataclass
class DataItem(Generic[T]):
    value: T
    dirty: bool

CorrelationStreamDict = Dict[int, Tuple[Tuple[int, int], np.ndarray]]

class RenderDataManager:
    def __init__(self) -> None:
        self.mutex: Lock = Lock()

        # Data storage
        self.light_image: Dict[int, DataItem[AvOutput]] = {}
        self.cam_image: Dict[int, DataItem[np.ndarray]] = {}
        self.depth_tracklets: Dict[int, DataItem[List[CamTracklet]]] = {}
        self.tracklets: Dict[int, DataItem[Tracklet]] = {}
        self.poses: Dict[int, DataItem[Pose]] = {}
        self.pose_streams: Dict[int, DataItem[PoseStreamData]] = {}
        self.r_streams: Dict[int, DataItem[PairCorrelationStreamData]] = {}

    def _set_data_dict(self, data_dict: Dict[int, DataItem[T]], key: int, value: T) -> None:
        with self.mutex:
            data_dict[key] = DataItem(value, True)

    def _get_data_dict(self, data_dict: Dict[int, DataItem[T]], key: int, use_dirty: bool = True) -> Optional[T]:
        with self.mutex:
            item: Optional[DataItem[T]] = data_dict.get(key)
            if not item:
                return None
            if use_dirty and not item.dirty:
                return None
            if use_dirty:
                item.dirty = False
            return item.value

    # Audio-visual data management
    def set_light_image(self, value: AvOutput) -> None:
        self._set_data_dict(self.light_image, 0, value)

    def get_light_image(self, only_if_dirty: bool = True) -> Optional[AvOutput]:
        return self._get_data_dict(self.light_image, 0, only_if_dirty)

    # Camera image management
    def set_cam_image(self, key: int, frame_type: FrameType, value: np.ndarray) -> None:
        self._set_data_dict(self.cam_image, key, value)

    def get_cam_image(self, key: int, only_if_dirty: bool = True) -> Optional[np.ndarray]:
        return self._get_data_dict(self.cam_image, key, only_if_dirty)

    # Depth tracklet management
    def set_depth_tracklets(self, key: int, value: list[CamTracklet]) -> None:
        self._set_data_dict(self.depth_tracklets, key, value)

    def get_depth_tracklets(self, key: int, only_if_dirty: bool = True) -> list[CamTracklet]:
        result: List[CamTracklet] | None = self._get_data_dict(self.depth_tracklets, key, only_if_dirty)
        return result if result is not None else []

    # Tracklet management
    def set_tracklet(self, value: Tracklet) -> None:
        self._set_data_dict(self.tracklets, value.id, value)

    def get_tracklet(self, id: int, only_if_dirty: bool = True) -> Optional[Tracklet]:
        return self._get_data_dict(self.tracklets, id, only_if_dirty)

    def get_tracklets(self) -> dict[int, Tracklet]:
        with self.mutex:
            return {k: v.value for k, v in self.tracklets.items() if v.value is not None}

    def get_tracklets_for_cam(self, cam_id: int) -> list[Tracklet]:
        with self.mutex:
            return [v.value for v in self.tracklets.values() if v.value is not None and v.value.cam_id == cam_id]

    # Pose management
    def set_pose(self, value: Pose) -> None:
        self._set_data_dict(self.poses, value.id, value)

    def get_pose(self, id: int, only_if_dirty: bool = True) -> Optional[Pose]:
        return self._get_data_dict(self.poses, id, only_if_dirty)

    def get_poses_for_cam(self, cam_id: int) -> List[Pose]:
        with self.mutex:
            return [v.value for v in self.poses.values() if v.value is not None and v.value.cam_id == cam_id]

    # Pose window/stream management
    def set_pose_stream(self, value: PoseStreamData) -> None:
        self._set_data_dict(self.pose_streams, value.id, value)

    def get_pose_stream(self, id: int, only_if_dirty: bool = True) -> Optional[PoseStreamData]:
        return self._get_data_dict(self.pose_streams, id, only_if_dirty)

    # Correlation window management
    def set_correlation_stream(self, value: PairCorrelationStreamData) -> None:
        self._set_data_dict(self.r_streams, 0, value)

    def get_correlation_streams(self, only_if_dirty: bool = True) -> Optional[PairCorrelationStreamData]:
        return self._get_data_dict(self.r_streams, 0, only_if_dirty)

