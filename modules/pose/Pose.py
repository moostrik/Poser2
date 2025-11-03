# Standard library imports
import numpy as np
from functools import cached_property
from dataclasses import dataclass, field, MISSING
from typing import Callable

from pandas import Timestamp

# Local application imports
from modules.pose.features.PosePoints import PosePointData
from modules.pose.features.PoseAngles import PoseAngleData, PoseAngleFactory
from modules.pose.features.PoseAngleSymmetry import PoseAngleSymmetryData, PoseAngleSymmetryFactory
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

    tracklet: Tracklet       # Deprecated, but kept for backward compatibility
    detection_image: np.ndarray # Cropped image corresponding to bounding_box (with padding)
    time_stamp: Timestamp    # Time when pose was captured -> should be Unix time in ms
    lost: bool               # Last frame, before being lost

    bounding_box: Rect       # Bounding Box, in normalized coordinates, can be outside [0,1]
    point_data: PosePointData
    angle_data: PoseAngleData
    angle_delta_data: PosePointData = field(default_factory=PosePointData.create_empty)

    def __repr__(self) -> str:
        return (f"Pose(id={self.tracklet.id}, points={self.point_data.valid_count if self.point_data else 0}, age={self.age:.2f}s)")

    @property
    def age(self) -> float:
        """Time in seconds since pose was captured"""
        return (Timestamp.now() - self.time_stamp).total_seconds()

    # LAZY FEATURES
    @cached_property
    def measurement_data(self) -> PoseMeasurementData:
        """Compute body measurements. Returns NaN values if point_data or crop_rect is None. Cached after first access."""
        return PoseMeasurementFactory.compute(self.point_data, self.bounding_box)

    @cached_property
    def similarity_data(self) -> PoseAngleSymmetryData:
        """Compute body measurements. Returns NaN values if point_data or crop_rect is None. Cached after first access."""
        return PoseAngleSymmetryFactory.from_angles(self.angle_data)

    # @cached_property
    # def absolute_points(self) -> np.ndarray:
    #     """Convert normalized keypoints to absolute pixel coordinates. Cached after first access."""
    #     pose_joints: np.ndarray = self.point_data.values
    #     rect: Rect = self.bounding_box

    #     # Vectorized conversion from normalized [0,1] to absolute pixel coordinates
    #     scale: np.ndarray = np.array([rect.width, rect.height])
    #     offset: np.ndarray = np.array([rect.x, rect.y])

    #     return pose_joints[:, :2] * scale + offset


PoseCallback = Callable[[Pose], None]
PoseDict = dict[int, Pose]
PoseDictCallback = Callable[[PoseDict], None]