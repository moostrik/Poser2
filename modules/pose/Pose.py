# Standard library imports
import numpy as np
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


@dataclass (frozen=True)
class Pose:
    # Set at first stage of pipeline
    tracklet: Tracklet = field(repr=False)

    # Set at second stage of pipeline
    crop_rect: Optional[Rect] = field(default = None)
    crop_image: Optional[np.ndarray] = field(default = None, repr=False)

    # Set at third stage of pipeline
    point_data: Optional[PosePointData] = field(default=None, repr=False)

    # Computed lazily
    _angle_data: Optional[PoseAngleData] = field(default=None)
    _head_data: Optional[PoseHeadData] = field(default=None)
    _measurement_data: Optional[PoseMeasurementData] = field(init=False, default=None, repr=False)
    _vertex_data: Optional[PoseVertexData] = field(init=False, default=None, repr=False)

    # LAZY FEATURES
    @property
    def angle_data(self) -> Optional[PoseAngleData]:
        if self._angle_data is None:
            object.__setattr__(self, '_angle_data', PoseAngles.compute(self.point_data))
        return self._angle_data

    @property
    def head_data(self) -> Optional[PoseHeadData]:
        if self._head_data is None and self.point_data is not None:
            object.__setattr__(self, '_head_data', PoseHead.compute(self.point_data))
        return self._head_data

    @property
    def measurement_data(self) -> Optional[PoseMeasurementData]:
        if self._measurement_data is None:
            object.__setattr__(self, '_measurement_data', PoseMeasurements.compute(self.point_data, self.crop_rect))
        return self._measurement_data

    @property
    def vertex_data(self) -> Optional[PoseVertexData]:
        if self._vertex_data is None:
            object.__setattr__(self, '_vertex_data', PoseVertices.compute_angled_vertices(self.point_data, self.angle_data))
        return self._vertex_data

    # COMFORT METHODS
    @property
    def absolute_points(self) -> Optional[np.ndarray]:
        """
        Get PoseJoints in the original rectangle coordinates.
        Returns a tuple of (PoseJoints, scores) or None if not available.
        """
        if self.point_data is None or self.crop_rect is None:
            return None

        PoseJoints: np.ndarray = self.point_data.points  # Normalized coordinates within the model
        rect: Rect = self.crop_rect

        # Convert from normalized coordinates to actual pixel coordinates in the crop rect
        real_PoseJoints: np.ndarray = np.zeros_like(PoseJoints)
        real_PoseJoints[:, 0] = PoseJoints[:, 0] * rect.width + rect.x  # x coordinates
        real_PoseJoints[:, 1] = PoseJoints[:, 1] * rect.height + rect.y  # y coordinates

        return real_PoseJoints

PoseCallback = Callable[[Pose], None]
PoseDict = dict[int, Pose]
PoseDictCallback = Callable[[PoseDict], None]