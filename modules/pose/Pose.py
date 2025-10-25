# Standard library imports
import numpy as np
from functools import cached_property
from dataclasses import dataclass, field
from typing import Optional, Callable

# Local application imports
from modules.pose.features.PosePoints import PosePointData
from modules.pose.features.PoseVertices import PoseVertices, PoseVertexData
from modules.pose.features.PoseAngles import PoseAngles, PoseAngleData
from modules.pose.features.PoseHeadOrientation import PoseHead, PoseHeadData
from modules.pose.features.PoseMeasurements import PoseMeasurements, PoseMeasurementData

from modules.tracker.Tracklet import Tracklet
from modules.utils.PointsAndRects import Rect


@dataclass(frozen=True)
class Pose:    
    """Immutable pose data structure with lazy feature computation.
    
    Designed for multi-stage pipeline:
    1. Detection: tracklet assigned
    2. Cropping: crop_rect and crop_image added
    3. Keypoint detection: point_data added
    4. Feature extraction: computed on-demand via cached properties
    
    All computed features return valid objects even when input data is missing,
    using NaN values to indicate unavailable data. This allows for consistent
    API without null checks, with validity determined by NaN presence.
    
    Only vertex_data and absolute_points return None when dependencies are unavailable.
    """
    
    # Set at first stage of pipeline
    tracklet: Tracklet = field(repr=False)

    # Set at second stage of pipeline
    crop_rect: Optional[Rect] = field(default=None)
    crop_image: Optional[np.ndarray] = field(default=None, repr=False)

    # Set at third stage of pipeline
    point_data: Optional[PosePointData] = field(default=None, repr=False)

    # LAZY FEATURES
    @cached_property
    def angle_data(self) -> PoseAngleData:
        """Compute joint angles. Returns NaN values if point_data is None. Cached after first access."""
        return PoseAngles.compute(self.point_data)

    @cached_property
    def head_data(self) -> PoseHeadData:
        """Compute head orientation. Returns NaN values if point_data is None. Cached after first access."""
        return PoseHead.compute(self.point_data)

    @cached_property
    def measurement_data(self) -> PoseMeasurementData:
        """Compute body measurements. Returns NaN values if point_data or crop_rect is None. Cached after first access."""
        return PoseMeasurements.compute(self.point_data, self.crop_rect)

    @cached_property
    def vertex_data(self) -> Optional[PoseVertexData]:
        """Compute skeleton vertices for rendering. Returns None if dependencies are unavailable. Cached after first access."""
        return PoseVertices.compute_angled_vertices(self.point_data, self.angle_data)

    @cached_property
    def absolute_points(self) -> Optional[np.ndarray]:
        """Convert normalized keypoints to absolute pixel coordinates. Returns None if point_data or crop_rect is None. Cached after first access."""
        if self.point_data is None or self.crop_rect is None:
            return None

        pose_joints: np.ndarray = self.point_data.points
        rect: Rect = self.crop_rect

        # Convert from normalized [0,1] to absolute pixel coordinates
        real_pose_joints: np.ndarray = np.zeros_like(pose_joints)
        real_pose_joints[:, 0] = pose_joints[:, 0] * rect.width + rect.x
        real_pose_joints[:, 1] = pose_joints[:, 1] * rect.height + rect.y

        return real_pose_joints


PoseCallback = Callable[[Pose], None]
PoseDict = dict[int, Pose]
PoseDictCallback = Callable[[PoseDict], None]