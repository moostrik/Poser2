from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Generic, TypeVar

import numpy as np

NORMALIZED_RANGE =  (0.0, 1.0)
POSITIVE_RANGE =    (0.0, np.inf)
UNBOUNDED_RANGE =   (-np.inf, np.inf)
PI_RANGE =          (0.0, np.pi)
TWO_PI_RANGE =      (0.0, 2 * np.pi)
SYMMETRIC_PI_RANGE= (-np.pi, np.pi)

FeatureEnum = TypeVar('FeatureEnum', bound=IntEnum)


class BaseFeature(ABC, Generic[FeatureEnum]):
    """Base interface for pose features.

    All pose features share:
    - Structure definition (joint enum)
    - Raw data arrays (values, scores)
    - Validity tracking
    - Common constructor pattern

    This enables polymorphic code that works with any feature type.

    Design Philosophy:
    ==================

    Raw Access (numpy-native):
    --------------------------
    feature.values      → Full array (shape varies by type)
    feature.scores      → Full scores (n_joints,)
    feature[joint]      → Single value (type varies by subclass)

    Use for: Numpy operations, batch processing, performance

    Python-Friendly Access:
    -----------------------
    feature.get(joint, fill)    → Python types (varies by subclass)
    feature.get_score(joint)    → Python float
    feature.get_scores(joints)  → Python list

    Use for: Logic, conditionals, unpacking, defaults
    """

    # ========== STRUCTURE ==========

    @classmethod
    @abstractmethod
    def joint_enum(cls) -> type[FeatureEnum]:
        """Joint enum defining the structure (source of truth for length).

        Returns:
            IntEnum subclass defining joints for this feature type.
        """
        pass

    @classmethod
    @abstractmethod
    def default_range(cls) -> tuple[float, float]:
        """Define valid value range. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Number of joints in this feature.

        Returns:
            Number of joints (typically len(joint_enum())).
        """
        pass

    # ========== RAW DATA ACCESS ==========

    @property
    @abstractmethod
    def values(self) -> np.ndarray:
        """Feature values array (read-only).

        Shape varies by feature type:
        - VectorFeature: (n_joints, n_dims)
        - ScalarFeature: (n_joints,)

        Use for raw numpy operations. For Python-friendly access,
        use get() methods defined by subclasses.

        Returns:
            Numpy array with feature values (NaN for invalid joints).
        """
        pass

    @property
    @abstractmethod
    def scores(self) -> np.ndarray:
        """Confidence scores array (read-only, n_joints).

        Raw numpy array for batch operations.
        For single values, use get_score() for Python float conversion.

        Returns:
            Array of confidence scores [0.0, 1.0] for each joint.
        """
        pass

    # ========== VALIDITY ==========

    @property
    @abstractmethod
    def valid_mask(self) -> np.ndarray:
        """Boolean mask of valid joints (n_joints,).

        Returns:
            Boolean array where True = valid data, False = invalid/NaN.
        """
        pass

    @property
    @abstractmethod
    def valid_count(self) -> int:
        """Number of joints with valid data.

        Returns:
            Count of True values in valid_mask.
        """
        pass

    # ========== SCORE ACCESS (Python-friendly) ==========

    @abstractmethod
    def get_score(self, joint: FeatureEnum | int) -> float:
        """Get confidence score for a single joint (Python float).

        Args:
            joint: Joint enum member or index

        Returns:
            Confidence score [0.0, 1.0] as Python float.
        """
        pass

    @abstractmethod
    def get_scores(self, joints: list[FeatureEnum | int]) -> list[float]:
        """Get confidence scores for multiple joints (Python list).

        Args:
            joints: List of joint enum members or indices

        Returns:
            List of confidence scores as Python floats.

        Note:
            For raw numpy access, use feature.scores[joints].
        """
        pass

    # ========== VALIDATION ==========

    @abstractmethod
    def validate(self, check_ranges: bool = True) -> tuple[bool, str | None]:
        """Validate feature data integrity.

        Args:
            check_ranges: Whether to perform range validation on values.

        Returns:
            Tuple of (is_valid, error_message if invalid).
        """
        pass

    # ========== REPRESENTATION ==========

    @abstractmethod
    def __repr__(self) -> str:
        """String representation for debugging.

        Returns:
            Human-readable string describing this feature.
        """
        pass

    # ========== CONSTRUCTION ==========
    # Note: While all features use __init__(values, scores), we don't
    # mark it as @abstractmethod because Python doesn't enforce
    # constructor signatures in subclasses. The pattern is documented here:
    #
    # Standard constructor pattern:
    #     def __init__(self, values: np.ndarray, scores: np.ndarray) -> None:
    #         """Initialize with raw values and confidence scores."""
    #         ...
    #
    # Standard factory methods:
    #     @classmethod
    #     def create_empty(cls) -> Self:
    #         """Create empty instance with all NaN values."""
    #
    #     @classmethod
    #     def from_values(cls, values: np.ndarray, scores: Optional[np.ndarray] = None) -> Self:
    #         """Create from values, generating default scores if needed."""
    #
    #     @classmethod
    #     def create_validated(cls, values: np.ndarray, scores: np.ndarray) -> Self:
    #         """Create with full validation."""