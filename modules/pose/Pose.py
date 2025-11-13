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

    is_removed: bool    # depricated, for use in stream

    time_stamp: float =         field(default_factory=time.time)
    bbox: BBoxFeature =         field(default_factory=BBoxFeature.create_dummy)
    points: Point2DFeature =    field(default_factory=Point2DFeature.create_dummy)
    angles: AngleFeature =      field(default_factory=AngleFeature.create_dummy)
    deltas: AngleFeature =      field(default_factory=AngleFeature.create_dummy)
    symmetry: SymmetryFeature = field(default_factory=SymmetryFeature.create_dummy)
    motion_time: float =        field(default=0.0)

    def __repr__(self) -> str:
        return (f"Pose(id={self.track_id}, points={self.points.valid_count}")

PoseCallback = Callable[[Pose], Any]
PoseDict = dict[int, Pose]
PoseDictCallback = Callable[[PoseDict], Any]