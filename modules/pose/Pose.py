# Standard library imports
from dataclasses import dataclass, field, fields
import time
from typing import Callable, Any

# Pose imports
from modules.pose.features import Point2DFeature, AngleFeature, SymmetryFeature, BBoxFeature
from enum import Enum

class PoseField(Enum):
    track_id = "track_id"
    cam_id = "cam_id"
    time_stamp = "time_stamp"
    bbox = "bbox"
    points = "points"
    angles = "angles"
    deltas = "deltas"
    symmetry = "symmetry"
    motion_time = "motion_time"

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

    def get_feature(self, feature: PoseField) -> Any:
        """Get a feature by its Enum value"""
        return getattr(self, feature.value)

    def __repr__(self) -> str:
        return (f"Pose(id={self.track_id}, points={self.points.valid_count}")

PoseCallback = Callable[[Pose], Any]
PoseDict = dict[int, Pose]
PoseDictCallback = Callable[[PoseDict], Any]


# Validation: Check that all enum values correspond to dataclass fields
_DEPRECATED_FIELDS = {"is_removed"}  # Fields that don't need enum entries

_pose_field_names: set[str] = {f.name for f in fields(Pose)} - _DEPRECATED_FIELDS
_enum_values: set[str] = {member.value for member in PoseField}
_missing_in_enum: set[str] = _pose_field_names - _enum_values
_missing_in_dataclass: set[str] = _enum_values - _pose_field_names

if _missing_in_enum:
    raise ValueError(f"Pose fields missing in PoseFeature enum: {_missing_in_enum}")
if _missing_in_dataclass:
    raise ValueError(f"PoseFeature enum values missing in Pose dataclass: {_missing_in_dataclass}")
