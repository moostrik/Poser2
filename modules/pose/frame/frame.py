# Standard library imports
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Any, TypeAlias, TYPE_CHECKING
import time

from ..features import Points2D, Angles, AngleVelocity, AngleMotion, AngleSymmetry, BBox, Similarity, LeaderScore, MotionGate

if TYPE_CHECKING:
    from .field import FrameField


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
        return (f"Frame(id={self.track_id}, points={self.points.valid_count})")


FrameCallback: TypeAlias =      Callable[[Frame], Any]
FrameDict: TypeAlias =          dict[int, Frame]
FrameDictCallback: TypeAlias =  Callable[[FrameDict], Any]
