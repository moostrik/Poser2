"""
Stores raw pose detection data at capture framerate.

Manages data storage for all detected entities at the rate they're captured from
cameras, enabling multiple consumers to access the same data independently.

Counterpart to RenderDataHub:
- CaptureDataHub: Stores raw pose data at capture rate (input FPS, e.g., 24 Hz)
- RenderDataHub: Provides smoothed pose data at render rate (output FPS, e.g., 60 Hz)

Key capabilities:
1. Thread-safe data storage with mutex protection
2. Per-consumer access tracking (new vs. already-accessed data)
3. Stores multi-modal data: images, tracklets, poses, correlations
4. Unique consumer key generation for independent data consumption

Data flow:
    Camera/Detector → CaptureDataHub.set_*() → CaptureDataHub.get_*() → Consumers
                                              ↓
                                       RenderDataHub (for smoothed output)

Consumer pattern:
    Each consumer gets a unique key to track which data they've already processed.
    This enables multiple independent consumers (e.g., UI, file writer, network sender)
    to consume the same data stream at their own pace.

Example:
    consumer_key = CaptureDataHub.get_unique_consumer_key()

    # First call returns data
    pose = hub.get_pose(tracklet_id, only_new_data=True, consumer_key=consumer_key)

    # Second call returns None (already consumed by this consumer)
    pose = hub.get_pose(tracklet_id, only_new_data=True, consumer_key=consumer_key)

    # New data arrives
    hub.set_poses(new_poses)

    # Now returns new data again
    pose = hub.get_pose(tracklet_id, only_new_data=True, consumer_key=consumer_key)
"""

from traceback import print_exc
import numpy as np
from typing import Optional, Callable
from threading import Lock
from dataclasses import dataclass, field
from typing import Optional, TypeVar, Generic


from modules.cam.depthcam.Definitions import Tracklet as DepthTracklet, FrameType
from modules.tracker.Tracklet import Tracklet, TrackletDict
from modules.pose.Pose import Pose, PoseDict
from modules.pose.similarity.Stream import StreamData
from modules.pose.features import AngleFeature, SimilarityBatch
from modules.WS.WSOutput import WSOutput

from modules.utils.HotReloadMethods import HotReloadMethods

T = TypeVar('T')

@dataclass
class DataItem(Generic[T]):
    value: T
    accessed: dict[str, bool] = field(default_factory=dict)

class DataHub:
    _consumer_counter: int = 0  # Class variable for unique IDs

    def __init__(self) -> None:
        self.mutex: Lock = Lock()

        # Data storage
        self.light_image: dict[int, DataItem[WSOutput]] = {}
        self.cam_image: dict[int, DataItem[np.ndarray]] = {}
        self.depth_tracklets: dict[int, DataItem[list[DepthTracklet]]] = {}
        self.tracklets: dict[int, DataItem[Tracklet]] = {}

        self.raw_poses: dict[int, DataItem[Pose]] = {}
        self.smooth_poses: dict[int, DataItem[Pose]] = {}
        self.interpolated_poses: dict[int, DataItem[Pose]] = {}

        self.pose_streams: dict[int, DataItem[StreamData]] = {}
        self.pose_correlation: dict[int, DataItem[SimilarityBatch ]] = {}
        self.motion_correlation: dict[int, DataItem[SimilarityBatch ]] = {}

        self._update_callback_lock = Lock()
        self._update_callbacks: set[Callable] = set()

        self._hot_reload = HotReloadMethods(self.__class__, True, True)

    @classmethod
    def get_unique_consumer_key(cls) -> str:
        """Generate a unique consumer key using a counter."""
        cls._consumer_counter += 1
        return f"C_{cls._consumer_counter}"

    def _set_data_dict(self, data_dict: dict[int, DataItem[T]], data_key: int, value: T) -> None:
        with self.mutex:
            data_dict[data_key] = DataItem(value)

    def _get_data_dict(self, data_dict: dict[int, DataItem[T]], data_key: int, only_new_data: bool, consumer_key: str) -> Optional[T]:
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
        result: list[DepthTracklet] | None = self._get_data_dict(self.depth_tracklets, key, only_new_data, consumer_key)
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
    def set_raw_poses(self, poses: PoseDict) -> None:
        for pose in poses.values():
            self._set_data_dict(self.raw_poses, pose.tracklet.id, pose)

    def get_raw_pose(self, id: int, only_new_data: bool, consumer_key: str) -> Optional[Pose]:
        return self._get_data_dict(self.raw_poses, id, only_new_data, consumer_key)

    def get_raw_poses_for_cam(self, cam_id: int) -> list[Pose]:
        with self.mutex:
            return [v.value for v in self.raw_poses.values() if v.value is not None and v.value.tracklet.cam_id == cam_id]

    def set_smooth_poses(self, poses: PoseDict) -> None:
        for pose in poses.values():
            self._set_data_dict(self.smooth_poses, pose.tracklet.id, pose)

    def get_smooth_pose(self, id: int, only_new_data: bool, consumer_key: str) -> Optional[Pose]:
        return self._get_data_dict(self.smooth_poses, id, only_new_data, consumer_key)

    def get_smooth_poses_for_cam(self, cam_id: int) -> list[Pose]:
        with self.mutex:
            return [v.value for v in self.smooth_poses.values() if v.value is not None and v.value.tracklet.cam_id == cam_id]

    def set_interpolated_poses(self, poses: PoseDict) -> None:
        for pose in poses.values():
            self._set_data_dict(self.interpolated_poses, pose.tracklet.id, pose)

    def get_interpolated_pose(self, id: int, only_new_data: bool, consumer_key: str) -> Optional[Pose]:
        return self._get_data_dict(self.interpolated_poses, id, only_new_data, consumer_key)

    def get_interpolated_poses_for_cam(self, cam_id: int) -> list[Pose]:
        with self.mutex:
            return [v.value for v in self.interpolated_poses.values() if v.value is not None and v.value.tracklet.cam_id == cam_id]

    # Pose window/stream management
    def set_pose_stream(self, value: StreamData) -> None:
        self._set_data_dict(self.pose_streams, value.id, value)

    def get_pose_stream(self, id: int, only_new_data: bool, consumer_key: str) -> Optional[StreamData]:
        return self._get_data_dict(self.pose_streams, id, only_new_data, consumer_key)

    # Correlation window management
    def set_pose_correlation(self, value: SimilarityBatch ) -> None:
        self._set_data_dict(self.pose_correlation, 0, value)

    def get_pose_correlation(self, only_new_data: bool, consumer_key: str) -> Optional[SimilarityBatch ]:
        return self._get_data_dict(self.pose_correlation, 0, only_new_data, consumer_key)

    def set_motion_correlation(self, value: SimilarityBatch ) -> None:
        self._set_data_dict(self.motion_correlation, 0, value)

    def get_motion_correlation(self, only_new_data: bool, consumer_key: str) -> Optional[SimilarityBatch ]:
        return self._get_data_dict(self.motion_correlation, 0, only_new_data, consumer_key)

    def get_is_active(self, tracklet_id: int) -> bool:
        with self.mutex:
            tracklet_item: Optional[DataItem[Tracklet]] = self.tracklets.get(tracklet_id)
            if not tracklet_item or not tracklet_item.value:
                return False
            return tracklet_item.value.is_being_tracked

    def get_angles(self, tracklet_id: int) -> Optional[AngleFeature]:
        with self.mutex:
            pose_item: Optional[DataItem[Pose]] = self.smooth_poses.get(tracklet_id)
            if not pose_item or not pose_item.value:
                return None
            return pose_item.value.angles

    # update callback
    def update(self) -> None:
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