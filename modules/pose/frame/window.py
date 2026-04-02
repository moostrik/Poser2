"""Window data types for frame feature buffering."""

from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Generic, TypeVar, TypeAlias, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from modules.pose.frame.field import FrameField

TEnum = TypeVar('TEnum', bound=IntEnum)


@dataclass(frozen=True)
class FeatureWindow(Generic[TEnum]):
    """Window of feature values with metadata.

    Shape: (time, feature_len) where time is oldest-first.

    Attributes:
        values: Feature values array of shape (time, feature_len)
        mask: Boolean validity mask of shape (time, feature_len)
        feature_enum: Enum class for feature elements (e.g., AngleLandmark)
        range: (min, max) tuple for validation range
        display_range: (min, max) tuple for visualization range
    """
    values: np.ndarray       # (time, feature_len)
    mask: np.ndarray         # (time, feature_len) bool
    feature_enum: type[TEnum]  # e.g., AngleLandmark
    range: tuple[float, float]  # (min, max) validation range
    display_range: tuple[float, float]  # (min, max) visualization range

    @property
    def feature_names(self) -> list[str]:
        """Derive feature names from enum."""
        return [e.name for e in self.feature_enum]

    def __getitem__(self, landmark: TEnum | int) -> np.ndarray:
        """Get time series for one feature element.

        Example: window[AngleLandmark.left_elbow] returns (time,) array
        """
        return self.values[:, landmark]

    @property
    def shape(self) -> tuple[int, ...]:
        """Return (time, feature_len) shape."""
        return self.values.shape

    @property
    def time_len(self) -> int:
        """Number of time samples in window."""
        return self.values.shape[0]

    @property
    def feature_len(self) -> int:
        """Number of feature elements."""
        return self.values.shape[1]


# {track_id: FeatureWindow}
FeatureWindowDict: TypeAlias =          dict[int, FeatureWindow]
FeatureWindowDictCallback: TypeAlias =  Callable[[FeatureWindowDict], None]

# {FrameField: FeatureWindowDict}  (field-first: each field maps to its per-track windows)
FrameWindowDict: TypeAlias =            dict['FrameField', FeatureWindowDict]
FrameWindowDictCallback: TypeAlias =    Callable[[FrameWindowDict], None]
