# Standard library imports
from dataclasses import dataclass, field
from functools import cached_property
import time
from typing import Callable, Any

# Third-party imports
import numpy as np

# Pose imports
from modules.pose.features import Point2DFeature, AngleFeature, SymmetryFeature, BBoxFeature
from modules.pose.features.deprecated.PoseMeasurements import PoseMeasurementData, PoseMeasurementFactory

# Local application imports
from modules.tracker.Tracklet import Tracklet
from modules.utils.PointsAndRects import Rect


@dataclass(frozen=True)
class Pose:
    """Immutable pose data structure"""
    track_id: int
    cam_id: int

    tracklet: Tracklet                              # Deprecated, but kept for backward compatibility
    crop_image: np.ndarray =    field(init=False)   # Deprecated Cropped image corresponding to bounding_box (with padding)
    lost: bool =                field(default=False) # Depricated, this is a tracking state, not a pose property

    time_stamp: float =         field(default_factory=time.time)
    bbox: BBoxFeature =         field(default_factory=BBoxFeature.create_dummy)
    points: Point2DFeature =    field(default_factory=Point2DFeature.create_dummy)
    angles: AngleFeature =      field(default_factory=AngleFeature.create_dummy)
    deltas: AngleFeature =      field(default_factory=AngleFeature.create_dummy)
    symmetry: SymmetryFeature = field(default_factory=SymmetryFeature.create_dummy)
    motion_time: float =        field(default=0.0)

    def __repr__(self) -> str:
        return (f"Pose(id={self.tracklet.id}, points={self.points.valid_count if self.points else 0}, age={self.age:.2f}s)")



    @property
    def age(self) -> float: # this also makes no sense
        return time.time() - self.time_stamp

    # LAZY FEATURES
    @cached_property
    def measurement_data(self) -> PoseMeasurementData:# deprecated
        return PoseMeasurementFactory.compute(self.points, self.bbox.to_rect())

    @cached_property
    def camera_points(self) -> Point2DFeature: # DEPRECATED
        pose_joints: np.ndarray = self.points.values
        rect: Rect = self.bbox.to_rect()

        # Vectorized conversion from normalized [0,1] to camera pixel coordinates
        scale: np.ndarray = np.array([rect.width, rect.height])
        offset: np.ndarray = np.array([rect.x, rect.y])

        camera_values = pose_joints[:, :2] * scale + offset

        return Point2DFeature(values=camera_values, scores=self.points.scores)


PoseCallback = Callable[[Pose], Any]
PoseDict = dict[int, Pose]
PoseDictCallback = Callable[[PoseDict], Any]