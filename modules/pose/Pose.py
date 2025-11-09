# Standard library imports
from dataclasses import dataclass, field
from functools import cached_property
import time
from typing import Callable

# Third-party imports
import numpy as np

# Pose imports
from modules.pose.features import Point2DFeature, PoseAngleData, PoseAngleSymmetryData
from modules.pose.features.deprecated.PoseMeasurements import PoseMeasurementData, PoseMeasurementFactory

# Local application imports
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
    crop_image: np.ndarray   # Cropped image corresponding to bounding_box (with padding)
    time_stamp: float        # Time when pose was captured
    lost: bool               # Last frame, before being lost

    bounding_box: Rect       # Bounding Box, in normalized coordinates, can be outside [0,1]
    point_data: Point2DFeature
    angle_data: PoseAngleData = field(default_factory=PoseAngleData.create_empty)
    delta_data: PoseAngleData = field(default_factory=PoseAngleData.create_empty)
    symmetry_data: PoseAngleSymmetryData = field(default_factory=PoseAngleSymmetryData.create_empty)
    motion_time: float =        0.0

    def __repr__(self) -> str:
        return (f"Pose(id={self.tracklet.id}, points={self.point_data.valid_count if self.point_data else 0}, age={self.age:.2f}s)")

    @property
    def age(self) -> float:
        """Time in seconds since pose was captured"""
        return time.time() - self.time_stamp

    # LAZY FEATURES
    @cached_property
    def measurement_data(self) -> PoseMeasurementData:
        """Compute body measurements. Returns NaN values if point_data or crop_rect is None. Cached after first access."""
        return PoseMeasurementFactory.compute(self.point_data, self.bounding_box)

    @cached_property
    def camera_points(self) -> Point2DFeature:
        """Convert normalized keypoints to camera image coordinates. Cached after first access."""
        pose_joints: np.ndarray = self.point_data.values
        rect: Rect = self.bounding_box

        # Vectorized conversion from normalized [0,1] to camera pixel coordinates
        scale: np.ndarray = np.array([rect.width, rect.height])
        offset: np.ndarray = np.array([rect.x, rect.y])

        camera_values = pose_joints[:, :2] * scale + offset

        # Preserve scores from original point_data
        return Point2DFeature(values=camera_values, scores=self.point_data.scores)


PoseCallback = Callable[[Pose], None]
PoseDict = dict[int, Pose]
PoseDictCallback = Callable[[PoseDict], None]