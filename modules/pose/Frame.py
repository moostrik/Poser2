# Standard library imports
from dataclasses import dataclass, field, fields
from enum import IntEnum, auto
import time
from typing import Callable, Any, get_type_hints

# Pose imports
from modules.pose.features import Points2D, Angles, AngleVelocity, AngleMotion, AngleSymmetry, BBox, Similarity, LeaderScore, MotionGate
from modules.pose.features.base import BaseFeature, BaseScalarFeature


@dataclass(frozen=True, slots=True)
class Frame:
    """Immutable pose data structure"""
    track_id: int
    cam_id: int

    # is_removed: bool    # depricated, for use in stream

    time_stamp: float =         field(default_factory=time.time)
    bbox: BBox =                field(default_factory=BBox.create_dummy)
    points: Points2D =          field(default_factory=Points2D.create_dummy)
    angles: Angles =            field(default_factory=Angles.create_dummy)
    angle_vel: AngleVelocity =  field(default_factory=AngleVelocity.create_dummy)
    angle_sym: AngleSymmetry =  field(default_factory=AngleSymmetry.create_dummy)
    angle_motion: AngleMotion = field(default_factory=AngleMotion.create_dummy)
    similarity: Similarity =    field(default_factory=Similarity.create_dummy)
    leader: LeaderScore =       field(default_factory=LeaderScore.create_dummy)
    motion_gate: MotionGate =   field(default_factory=MotionGate.create_dummy)
    motion_time: float =        field(default=0.0)
    age: float =                field(default=0.0)
    model_ar: float =           field(default=0.75)

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
    angle_motion =  auto()
    angle_sym =     auto()
    similarity =    auto()
    leader =        auto()
    motion_gate =   auto()
    motion_time =   auto()
    age =           auto()
    model_ar =      auto()

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
            return feature_type.range()
        raise ValueError(
            f"PoseField '{self.name}' of type '{feature_type.__name__}' does not have a defined range"
        )

    def get_display_range(self) -> tuple[float, float]:
        """Get the display range for this feature, if applicable."""
        feature_type: type = self.get_type()
        if issubclass(feature_type, BaseFeature):
            return feature_type.display_range()
        raise ValueError(
            f"PoseField '{self.name}' of type '{feature_type.__name__}' does not have a defined display range"
        )

    def get_length(self) -> int:
        """Get the length of this feature, if applicable."""
        feature_type: type = self.get_type()
        if issubclass(feature_type, BaseFeature):
            enum_cls = feature_type.enum()
            if enum_cls is not None and isinstance(enum_cls, type) and issubclass(enum_cls, IntEnum):
                return len(enum_cls)
            raise ValueError(
                f"{feature_type.__name__}.enum() did not return an IntEnum class"
            )
        raise ValueError(
            f"PoseField '{self.name}' of type '{feature_type.__name__}' does not have a defined length"
        )

    @staticmethod
    def get_scalar_fields() -> list['FrameField']:
        """Return all FrameField members whose type is a BaseScalarFeature subclass.

        Excludes Points2D (BaseVectorFeature) and plain scalars (float, int, bool).
        This is the single source of truth for which fields can be windowed.
        """
        result: list[FrameField] = []
        for ff in FrameField:
            try:
                ft = ff.get_type()
                if isinstance(ft, type) and issubclass(ft, BaseScalarFeature):
                    result.append(ff)
            except (ValueError, TypeError):
                pass
        return result


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


class ScalarFrameField(IntEnum):
    """Subset of FrameField containing only BaseScalarFeature fields.

    Used for data visualization where only scalar features are valid.
    Values match FrameField so they work interchangeably as dict keys
    and with Frame.get_feature() which uses .name attribute.
    """
    bbox =          FrameField.bbox
    angles =        FrameField.angles
    angle_vel =     FrameField.angle_vel
    angle_motion =  FrameField.angle_motion
    angle_sym =     FrameField.angle_sym
    similarity =    FrameField.similarity
    leader =        FrameField.leader
    motion_gate =   FrameField.motion_gate


# SCALAR FRAME FIELD VALIDATION
_scalar_names = {ff.name for ff in FrameField.get_scalar_fields()}
_enum_scalar_names = {sf.name for sf in ScalarFrameField}
_missing_in_scalar_enum = _scalar_names - _enum_scalar_names
_extra_in_scalar_enum = _enum_scalar_names - _scalar_names

if _missing_in_scalar_enum:
    raise ValueError(f"ScalarFrameField missing scalar fields: {_missing_in_scalar_enum}")
if _extra_in_scalar_enum:
    raise ValueError(f"ScalarFrameField has non-scalar fields: {_extra_in_scalar_enum}")
