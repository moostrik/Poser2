from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Generic, TypeVar

import numpy as np


FeatureEnum = TypeVar('FeatureEnum', bound=IntEnum)


class BaseFeature(ABC, Generic[FeatureEnum]):
    """Minimal shared interface for pose features.

    All features represent per-joint data with confidence scores.
    Subclasses define the specific data format (scalars, points, etc.)
    """

    # ========== STRUCTURE ==========

    @classmethod
    @abstractmethod
    def joint_enum(cls) -> type[FeatureEnum]:
        """Joint enum defining the structure (source of truth for length)."""
        pass

    def __len__(self) -> int:
        """Number of joints."""
        return len(self.joint_enum())

    # ========== VALIDITY ==========

    @property
    @abstractmethod
    def scores(self) -> np.ndarray:
        """Confidence scores (n_joints,)."""
        pass

    @property
    @abstractmethod
    def valid_mask(self) -> np.ndarray:
        """Boolean mask of valid joints (n_joints,)."""
        pass

    @property
    @abstractmethod
    def valid_count(self) -> int:
        """Number of valid joints."""
        pass

    # ========== VALIDATION ==========

    @abstractmethod
    def validate(self, check_ranges: bool = True) -> tuple[bool, str | None]:
        """Validate feature data (implementation varies by type)."""
        pass

    # ========== REPRESENTATION ==========

    @abstractmethod
    def __repr__(self) -> str:
        """String representation."""
        pass