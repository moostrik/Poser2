# Standard library imports
from dataclasses import dataclass, field, fields
from enum import IntEnum, auto
import time
from typing import Callable, Any, get_type_hints

# Pose imports
from modules.pose.features import Points2D, Angles, AngleVelocity, AngleSymmetry, BBox, Similarity
from modules.pose.features.base import BaseFeature


@dataclass(frozen=True)
class Frame:
    """Immutable pose data structure"""
    track_id: int
    cam_id: int

    is_removed: bool    # depricated, for use in stream

    time_stamp: float =         field(default_factory=time.time)
    bbox: BBox =                field(default_factory=BBox.create_dummy)
    points: Points2D =          field(default_factory=Points2D.create_dummy)
    angles: Angles =            field(default_factory=Angles.create_dummy)
    angle_vel: AngleVelocity =  field(default_factory=AngleVelocity.create_dummy)
    angle_sym: AngleSymmetry =  field(default_factory=AngleSymmetry.create_dummy)
    similarity: Similarity =    field(default_factory=Similarity.create_dummy)
    motion_time: float =        field(default=0.0)
    age: float =                field(default=0.0)

    def get_feature(self, feature: 'FrameField') -> Any:
        """Get a feature by its Enum value"""
        return getattr(self, feature.name)

    def __repr__(self) -> str:
        return (f"Pose(id={self.track_id}, points={self.points.valid_count}")


FrameCallback = Callable[[Frame], Any]
FrameDict = dict[int, Frame]
FrameDictCallback = Callable[[FrameDict], Any]


class FrameField(IntEnum):
    track_id =      0
    cam_id =        auto()
    time_stamp =    auto()
    bbox =          auto()
    points =        auto()
    angles =        auto()
    angle_vel =     auto()
    angle_sym =     auto()
    similarity =    auto()
    motion_time =   auto()
    age =           auto()

    def get_type(self) -> type:
        """Get the feature class for this pose field using Pose type hints."""
        pose_type_hints = get_type_hints(Frame)
        if self.name not in pose_type_hints:
            raise ValueError(f"PoseField '{self.name}' is not a feature field")
        return pose_type_hints[self.name]

    def get_range(self) -> tuple[float, float]:
        """Get the valid range for this feature, if applicable."""
        feature_type: type = self.get_type()
        if issubclass(feature_type, BaseFeature):
            return feature_type.default_range()
        raise ValueError(
            f"PoseField '{self.name}' of type '{feature_type.__name__}' does not have a defined range"
        )

    def get_length(self) -> int:
        """Get the length of this feature, if applicable."""
        feature_type: type = self.get_type()
        if issubclass(feature_type, BaseFeature):
            enum_cls = feature_type.feature_enum()
            if enum_cls is not None and isinstance(enum_cls, type) and issubclass(enum_cls, IntEnum):
                return len(enum_cls)
            raise ValueError(
                f"{feature_type.__name__}.feature_enum() did not return an IntEnum class"
            )
        raise ValueError(
            f"PoseField '{self.name}' of type '{feature_type.__name__}' does not have a defined length"
        )



# FRAME FIELD VALIDATION
_DEPRECATED_FIELDS: set[str] = {"is_removed"}  # Fields that don't need enum entries

_pose_field_names: set[str] = {f.name for f in fields(Frame)} - _DEPRECATED_FIELDS
_enum_names: set[str] = {member.name for member in FrameField}
_missing_in_enum: set[str] = _pose_field_names - _enum_names
_missing_in_dataclass: set[str] = _enum_names - _pose_field_names

if _missing_in_enum:
    raise ValueError(f"Pose fields missing in PoseField enum: {_missing_in_enum}")
if _missing_in_dataclass:
    raise ValueError(f"PoseField enum values missing in Pose dataclass: {_missing_in_dataclass}")
