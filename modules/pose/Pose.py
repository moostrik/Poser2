# Standard library imports
import numpy as np
from functools import cached_property
from dataclasses import dataclass, field
from typing import Optional, Callable

from pandas import Timestamp

# Local application imports
from modules.pose.features.PosePoints import PosePointData
from modules.pose.features.PoseVertices import PoseVertexData, PoseVertexFactory
from modules.pose.features.PoseAngles import PoseAngleData, PoseAngleFactory
from modules.pose.features.depricated.PoseHeadOrientation import PoseHeadData,PoseHeadFactory
from modules.pose.features.PoseMeasurements import PoseMeasurementData, PoseMeasurementFactory

from modules.tracker.Tracklet import Tracklet
from modules.utils.PointsAndRects import Rect


@dataclass(frozen=True)
class Pose:
    """Immutable pose data structure with lazy feature computation.

    Designed for multi-stage pipeline:
    1. Input: tracklet, time_stamp, crop_rect and crop_image added
    2. Keypoint detection: point_data added
    3. Feature extraction: computed on-demand via cached properties

    All computed features return valid objects even when input data is missing,
    using NaN values to indicate unavailable data. This allows for consistent
    API without null checks, with validity determined by NaN presence.

    Only vertex_data and absolute_points return None when dependencies are unavailable.
    """

    # Set at first stage of pipeline
    tracklet: Tracklet = field(init=True)
    crop_rect: Rect = field(init=True)
    crop_image: np.ndarray = field(init=True)
    time_stamp: Timestamp = field(init=True)

    # Set at second stage of pipeline
    point_data: Optional[PosePointData] = field(default=None)

    def __repr__(self) -> str:
        return (f"Pose(id={self.tracklet.id}, valid={self.has_data}, "
                f"points={self.point_data.valid_count if self.point_data else 0}, "
                f"age={self.age:.2f}s)")

    @property
    def has_data(self) -> bool:
        """Check if pose has minimum required data for feature extraction"""
        return self.point_data is not None

    @property
    def age(self) -> float:
        """Time in seconds since pose was captured"""
        return (Timestamp.now() - self.time_stamp).total_seconds()

    # LAZY FEATURES
    @cached_property
    def angle_data(self) -> PoseAngleData:
        """Compute joint angles. Returns NaN values if point_data is None. Cached after first access."""
        return PoseAngleFactory.from_points(self.point_data)

    @cached_property
    def head_data(self) -> PoseHeadData:
        """Compute head orientation. Returns NaN values if point_data is None. Cached after first access."""
        return PoseHeadFactory.from_points(self.point_data)

    @cached_property
    def measurement_data(self) -> PoseMeasurementData:
        """Compute body measurements. Returns NaN values if point_data or crop_rect is None. Cached after first access."""
        return PoseMeasurementFactory.compute(self.point_data, self.crop_rect)

    @cached_property
    def vertex_data(self) -> Optional[PoseVertexData]:
        """Compute skeleton vertices for rendering. Returns None if dependencies are unavailable. Cached after first access."""
        return PoseVertexFactory.compute_angled_vertices(self.point_data, self.angle_data)

    @cached_property
    def absolute_points(self) -> Optional[np.ndarray]:
        """Convert normalized keypoints to absolute pixel coordinates. Returns None if point_data or crop_rect is None. Cached after first access."""
        if self.point_data is None or self.crop_rect is None:
            return None

        pose_joints: np.ndarray = self.point_data.values
        rect: Rect = self.crop_rect

        # Convert from normalized [0,1] to absolute pixel coordinates
        real_pose_joints: np.ndarray = np.zeros_like(pose_joints)
        real_pose_joints[:, 0] = pose_joints[:, 0] * rect.width + rect.x
        real_pose_joints[:, 1] = pose_joints[:, 1] * rect.height + rect.y

        return real_pose_joints


PoseCallback = Callable[[Pose], None]
PoseDict = dict[int, Pose]
PoseDictCallback = Callable[[PoseDict], None]