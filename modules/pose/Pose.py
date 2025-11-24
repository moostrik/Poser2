# Standard library imports
from dataclasses import dataclass, field, fields
import time
from typing import Callable, Any, Type

# Pose imports
from modules.pose.features import Points2D, Angles, Symmetry, BBox
from modules.pose.features.base import BaseFeature, BaseScalarFeature, BaseVectorFeature, NormalizedScalarFeature
from enum import Enum


@dataclass(frozen=True)
class Pose:
    """Immutable pose data structure"""
    track_id: int
    cam_id: int

    is_removed: bool    # depricated, for use in stream

    time_stamp: float =     field(default_factory=time.time)
    bbox: BBox =            field(default_factory=BBox.create_dummy)
    points: Points2D =      field(default_factory=Points2D.create_dummy)
    angles: Angles =        field(default_factory=Angles.create_dummy)
    deltas: Angles =        field(default_factory=Angles.create_dummy)
    symmetry: Symmetry =    field(default_factory=Symmetry.create_dummy)
    motion_time: float =    field(default=0.0)

    def get_feature(self, feature: 'PoseField') -> Any:
        """Get a feature by its Enum value"""
        return getattr(self, feature.value)

    def __repr__(self) -> str:
        return (f"Pose(id={self.track_id}, points={self.points.valid_count}")


PoseCallback = Callable[[Pose], Any]
PoseDict = dict[int, Pose]
PoseDictCallback = Callable[[PoseDict], Any]


class PoseField(Enum):
    track_id =      "track_id"
    cam_id =        "cam_id"
    time_stamp =    "time_stamp"
    bbox =          "bbox"
    points =        "points"
    angles =        "angles"
    deltas =        "deltas"
    symmetry =      "symmetry"
    motion_time =   "motion_time"

    def get_feature_class(self) -> Type[BaseFeature]:
        """Get the feature class for this pose field.

        Raises:
            ValueError: If this field is not a feature field
        """
        _FEATURE_MAP: dict[PoseField, Type[BaseFeature]] = {
            PoseField.bbox: BBox,
            PoseField.points: Points2D,
            PoseField.angles: Angles,
            PoseField.deltas: Angles,
            PoseField.symmetry: Symmetry,
        }

        if self not in _FEATURE_MAP:
            raise ValueError(f"PoseField '{self.value}' is not a feature field")
        return _FEATURE_MAP[self]

    def is_feature(self) -> bool:
        """Check if this is a feature field (vs metadata/timestamp)."""
        try:
            self.get_feature_class()
            return True
        except ValueError:
            return False

    def is_scalar_feature(self) -> bool:
        """Check if this is a scalar feature field."""
        feature_class = self.get_feature_class()
        return issubclass(feature_class, BaseScalarFeature)

    def is_vector_feature(self) -> bool:
        """Check if this is a vector feature field."""
        feature_class = self.get_feature_class()
        return issubclass(feature_class, BaseVectorFeature)

    def is_normalized_scalar_feature(self) -> bool:
        """Check if this is a normalized scalar feature field."""
        feature_class = self.get_feature_class()
        return issubclass(feature_class, NormalizedScalarFeature)

    def is_point_feature(self) -> bool:
        """Check if this is a 2D point feature field."""
        feature_class = self.get_feature_class()
        return issubclass(feature_class, Points2D)

    def is_angle_feature(self) -> bool:
        """Check if this is an angle feature field."""
        feature_class = self.get_feature_class()
        return issubclass(feature_class, Angles)


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
