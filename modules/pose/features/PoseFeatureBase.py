from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from typing import Generic, TypeVar
from enum import Enum, IntEnum
import numpy as np

from typing_extensions import Self

JointEnum = TypeVar('JointEnum', bound=IntEnum)

class FeatureStatistic(Enum):
    """Available statistical metrics for pose features."""
    MEAN = 'mean'
    GEOMETRIC_MEAN = 'geometric_mean'
    HARMONIC_MEAN = 'harmonic_mean'
    MIN = 'min_value'
    MAX = 'max_value'
    STD = 'std'
    MEDIAN = 'median'

@dataclass(frozen=True)
class PoseFeatureBase(ABC, Generic[JointEnum]):
    """Abstract base class for per-joint pose features with confidence scores.

    Immutable container for per-joint measurements with confidence scores.
    Provides validation, statistics, and convenient access patterns.
    """

    values: np.ndarray
    scores: np.ndarray

    def __post_init__(self) -> None:
        """Freeze arrays and validate data integrity.

        This is the ONLY __post_init__ - subclasses should NOT override it.
        Use validate() for custom validation logic.
        """
        # Make arrays immutable
        self.values.flags.writeable = False
        self.scores.flags.writeable = False

        # Run all validations
        self._validate_base()
        self.validate()  # Hook for subclass validation

    def _validate_base(self) -> None:
        """Base class validation - DO NOT OVERRIDE.

        Validates:
        - Arrays are 1D with correct length for joint enum
        - NaN values must have 0.0 scores
        """
        # Validate both arrays are 1D
        if self.values.ndim != 1:
            raise ValueError(
                f"values array must be 1D, got shape {self.values.shape}"
            )
        if self.scores.ndim != 1:
            raise ValueError(
                f"scores array must be 1D, got shape {self.scores.shape}"
            )

        # Validate array length matches joint enum - cache the enum type
        joint_enum_type: type[JointEnum] = self.__class__.joint_enum()
        expected_length: int = len(joint_enum_type)
        if len(self.values) != expected_length:
            raise ValueError(
                f"values array length ({len(self.values)}) must match joint enum length ({expected_length})"
            )
        if len(self.scores) != expected_length:
            raise ValueError(
                f"scores array length ({len(self.scores)}) must match joint enum length ({expected_length})"
            )

        # Validate: NaN values must have 0.0 scores
        nan_mask = np.isnan(self.values)
        valid_mask = self.scores > 0.0
        invalid = nan_mask & valid_mask

        if np.any(invalid):
            invalid_joints = [joint_enum_type(i).name for i in np.where(invalid)[0]]
            raise ValueError(
                f"Data integrity violation: NaN values must have 0.0 scores. "
                f"Invalid joints: {', '.join(invalid_joints)}"
            )

    @abstractmethod
    def validate(self) -> None:
        """Custom validation logic for subclasses.

        Override this method to add subclass-specific validation.
        Called automatically during __post_init__.

        Raises:
            ValueError: If validation fails
        """
        pass  # Default: no additional validation

    @classmethod
    @abstractmethod
    def joint_enum(cls) -> type[JointEnum]:
        """Get the joint enum type for this class. Must be implemented by subclasses."""
        pass

    @classmethod
    def from_values(cls, values: np.ndarray) -> Self:
        """Create feature data from values array, auto-generating scores, 1.0 for valid (non-NaN) values, 0.0 for NaN values."""
        scores: np.ndarray = np.where(~np.isnan(values), 1.0, 0.0).astype(np.float32)
        return cls(values=values, scores=scores)

    @classmethod
    def create_empty(cls) -> Self:
        """Create instance with all joints marked as invalid (NaN values, zero scores)."""
        num_joints: int = len(cls.joint_enum())
        values: np.ndarray = np.full(num_joints, np.nan, dtype=np.float32)
        scores: np.ndarray = np.zeros(num_joints, dtype=np.float32)
        return cls(values=values, scores=scores)

    def __repr__(self) -> str:
        """Readable string representation."""
        if not self.any_valid:
            return f"{self.__class__.__name__}(0/{len(self.values)})"

        mean_score = float(np.mean(self.scores[self.valid_mask]))
        return f"{self.__class__.__name__}({self.valid_count}/{len(self.values)}, score={mean_score:.2f})"
        # ========== VALIDATION ==========

    @cached_property
    def valid_mask(self) -> np.ndarray:
        """Boolean array indicating which joints have valid (non-zero score) values."""
        return self.scores > 0.0

    @cached_property
    def valid_count(self) -> int:
        """Number of joints with valid (non-zero score) values."""
        return int(np.sum(self.valid_mask))

    @cached_property
    def any_valid(self) -> bool:
        """True if at least one valid value is available."""
        return self.valid_count > 0

    @cached_property
    def valid_values(self) -> np.ndarray:
        """Array of valid (non-zero score) values."""
        return self.values[self.valid_mask]

    @cached_property
    def valid_joints(self) -> list[JointEnum]:
        """List of joints with valid (non-zero score) values."""
        return [self.joint_enum()(i) for i in range(len(self.values)) if self.scores[i] > 0.0]

    # ========== STATISTICS ==========

    @cached_property
    def mean(self) -> float:
        """Mean of valid values, or NaN if none are valid."""
        return float(np.nanmean(self.values))

    @cached_property
    def harmonic_mean(self) -> float:
        """Harmonic mean of valid values, or NaN if none are valid."""
        valid = self.values[self.valid_mask]
        if valid.size == 0:
            return np.nan
        return float(valid.size / np.sum(1.0 / np.maximum(valid, 1e-10)))

    @cached_property
    def geometric_mean(self) -> float:
        """Geometric mean of valid values, or NaN if none are valid."""
        valid = self.values[self.valid_mask]
        if valid.size == 0:
            return np.nan
        return float(np.exp(np.mean(np.log(np.maximum(valid, 1e-10)))))

    @cached_property
    def min_value(self) -> float:
        """Minimum of valid values, or NaN if none are valid."""
        return float(np.nanmin(self.values))

    @cached_property
    def max_value(self) -> float:
        """Maximum of valid values, or NaN if none are valid."""
        return float(np.nanmax(self.values))

    @cached_property
    def std(self) -> float:
        """Standard deviation of valid values, or NaN if none are valid."""
        return float(np.nanstd(self.values))

    @cached_property
    def median(self) -> float:
        """Median of valid values, or NaN if none are valid."""
        return float(np.nanmedian(self.values))

    def get_stat(self, statistic: FeatureStatistic) -> float:
        """Get the value for a specific statistical metric, can be NaN."""
        return getattr(self, statistic.value)

    # ========== ACCESS ==========

    def __len__(self) -> int:
        """Return total number of joints (including invalid)."""
        return len(self.values)

    def __getitem__(self, joint: JointEnum | int) -> float:
        """Get value for a joint (may be NaN)."""
        return float(self.values[joint])

    def get(self, joint: JointEnum | int, default: float = np.nan) -> float:
        """Get value with default for NaN."""
        value = self.values[joint]
        return value if not np.isnan(value) else default

    def get_score(self, joint: JointEnum | int) -> float:
        """Get score for a joint (always between 0.0 and 1.0)."""
        return self.scores[joint]

    def to_dict(self, include_invalid: bool = False) -> dict[str, float]:
        """Convert to dictionary.

        Args:
            include_invalid: If True, includes NaN values. If False, only valid joints.

        Returns:
            Dictionary mapping joint names to values
        """
        if include_invalid:
            return {self.joint_enum()(i).name: float(v) for i, v in enumerate(self.values)}
        else:
            return {joint.name: self.values[joint] for joint in self.valid_joints}
