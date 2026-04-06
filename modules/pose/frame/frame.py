"""
=============================================================================
FRAME — ECS-LITE POSE DATA STRUCTURE
=============================================================================

Immutable container for pose data with 3 identity slots + a feature dict
keyed by feature class. Each app puts only its own features on Frame.

Identity (fixed slots):
  • track_id: int          — Tracked object ID
  • cam_id: int            — Camera source ID
  • time_stamp: float      — Capture timestamp (default: time.time())

Feature Access:
  • frame[FeatureType]           → feature instance (KeyError if absent)
  • frame.get(FeatureType)       → feature instance or None
  • FeatureType in frame         → membership check
  • len(frame)                   → number of features stored

Mutation (functional — returns new Frame):
  • replace(frame, {Angles: new_angles, ...})  → new Frame with merged features

Immutability:
  • __setattr__ / __delattr__ raise AttributeError
  • _features dict is shallow-copied at construction
  • Thread-safe: shared between input and render threads

Construction:
  • Frame(track_id, cam_id)
  • Frame(track_id, cam_id, time_stamp=t, features={Angles: angles, BBox: bbox})

Feature Authoring Guide:
========================

To add a new feature to the system:

  1. Create a subclass of BaseScalarFeature or BaseVectorFeature in
     modules/pose/features/. Implement enum(), range(), display_range().

  2. Export it from modules/pose/features/__init__.py.

  3. Write an extractor node (NodeBase subclass) that reads upstream features
     from Frame and writes the new feature via replace():
       data = MyFeature.from_value(computed_value)
       return replace(frame, {MyFeature: data})

  4. Wire the extractor into the app's FilterTracker pipeline
     (e.g. pose_smooth_filters in main.py).

  5. If the feature should be windowed, add it to the app's
     scalar_features list passed to FrameWindowTracker.

  6. If the feature needs UI selection in data layers, add a member
     to ScalarFeatureSelect and FEATURE_MAP in DataLayerSettings.

  No Frame class changes are needed — any BaseFeature subclass can be
  stored on Frame without modification (ECS-Lite principle).
=============================================================================
"""
from __future__ import annotations
from typing import Callable, Any, TypeAlias, TypeVar, TYPE_CHECKING
import time

if TYPE_CHECKING:
    from modules.pose.features.base import BaseFeature

T = TypeVar('T')


class Frame:
    """Immutable pose data container — identity slots + feature dict."""

    __slots__ = ('_track_id', '_cam_id', '_time_stamp', '_features')

    def __init__(
        self,
        track_id: int,
        cam_id: int,
        time_stamp: float | None = None,
        features: dict[type[BaseFeature], BaseFeature] | None = None,
    ) -> None:
        object.__setattr__(self, '_track_id', track_id)
        object.__setattr__(self, '_cam_id', cam_id)
        object.__setattr__(self, '_time_stamp', time_stamp if time_stamp is not None else time.time())
        object.__setattr__(self, '_features', dict(features) if features else {})

    # ========== IDENTITY ==========

    @property
    def track_id(self) -> int:
        return self._track_id

    @property
    def cam_id(self) -> int:
        return self._cam_id

    @property
    def time_stamp(self) -> float:
        return self._time_stamp

    # ========== FEATURE ACCESS ==========

    def __getitem__(self, feature_type: type[T]) -> T:
        """Get feature by type. Raises KeyError if absent."""
        try:
            return self._features[feature_type]
        except KeyError:
            raise KeyError(f"Frame has no {feature_type.__name__} feature")

    def get(self, feature_type: type[T], default: T | None = None) -> T | None:
        """Get feature by type, or default if absent."""
        return self._features.get(feature_type, default)

    def __contains__(self, feature_type: type) -> bool:
        """Check if a feature type is present."""
        return feature_type in self._features

    def __len__(self) -> int:
        """Number of features stored."""
        return len(self._features)

    # ========== IMMUTABILITY ==========

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Frame is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Frame is immutable")

    # ========== REPRESENTATION ==========

    def __repr__(self) -> str:
        feature_names = ', '.join(t.__name__ for t in self._features)
        return f"Frame(id={self._track_id}, features=[{feature_names}])"


def replace(frame: Frame, updates: dict[type[BaseFeature], BaseFeature]) -> Frame:
    """Create a new Frame with merged features (same identity).

    Args:
        frame: Source frame.
        updates: Feature dict to merge on top of existing features.

    Returns:
        New Frame with same identity and merged features.
    """
    merged = dict(frame._features)
    merged.update(updates)
    return Frame(
        track_id=frame._track_id,
        cam_id=frame._cam_id,
        time_stamp=frame._time_stamp,
        features=merged,
    )


FrameCallback: TypeAlias =      Callable[[Frame], Any]
FrameDict: TypeAlias =          dict[int, Frame]
FrameDictCallback: TypeAlias =  Callable[[FrameDict], Any]
