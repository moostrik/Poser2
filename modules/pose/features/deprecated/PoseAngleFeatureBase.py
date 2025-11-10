# Standard library imports
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, IntEnum
from functools import cached_property
from typing import Generic, TypeVar

# Third-party imports
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
class PoseAngleFeatureBase(ABC, Generic[JointEnum]):
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

        # Validate similarity values are in default range
        valid_values = self.values[~np.isnan(self.values)]
        if valid_values.size > 0:
            min_range, max_range = self.__class__.default_range()
            if np.any((valid_values < min_range) | (valid_values > max_range)):
                out_of_range = valid_values[(valid_values < min_range) | (valid_values > max_range)]
                raise ValueError(
                    f"Values must be in range [{min_range}, {max_range}]. "
                    f"Found values outside range: {out_of_range}"
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

    # ========== ABSTRACT METHODS ==========

    @classmethod
    @abstractmethod
    def joint_enum(cls) -> type[JointEnum]:
        """Get the joint enum type for this class. Must be implemented by subclasses."""
        pass

    @classmethod
    @abstractmethod
    def default_range(cls) -> tuple[float, float]:
        """Get the default/expected range for this feature type."""
        pass

    # ========== CONSTRUCTORS ==========

    @classmethod
    def from_values(cls, values: np.ndarray, scores: np.ndarray | None = None) -> Self:
        """Create feature data from values array, auto-generating scores, 1.0 for valid (non-NaN) values, 0.0 for NaN values."""
        if scores is None:
            scores = np.where(~np.isnan(values), 1.0, 0.0).astype(np.float32)
        return cls(values=values, scores=scores)

    @classmethod
    def create_empty(cls) -> Self:
        """Create instance with all joints marked as invalid (NaN values, zero scores)."""
        num_joints: int = len(cls.joint_enum())
        values: np.ndarray = np.full(num_joints, np.nan, dtype=np.float32)
        scores: np.ndarray = np.zeros(num_joints, dtype=np.float32)
        return cls(values=values, scores=scores)

    # ========== REPRESENTATION ==========

    def __repr__(self) -> str:
        """Readable string representation."""
        if not self.any_valid:
            return f"{self.__class__.__name__}(0/{len(self.values)})"

        mean_score = float(np.mean(self.scores[self.valid_mask]))
        return f"{self.__class__.__name__}({self.valid_count}/{len(self.values)}, score={mean_score:.2f})"

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
        joint_enum_type: type[JointEnum] = self.joint_enum()
        return [joint_enum_type(i) for i in np.where(self.valid_mask)[0]]

    # ========== STATISTICS ==========

    @cached_property
    def mean(self) -> float:
        """Mean of valid values, or NaN if none are valid."""
        return float(np.nanmean(self.values))

    @cached_property
    def harmonic_mean(self) -> float:
        """Harmonic mean of valid positive values, or NaN if none exist.

        Only meaningful for features with positive values (e.g., symmetry scores).
        Returns NaN if no valid positive values exist.
        """
        valid: np.ndarray = self.values[self.valid_mask]
        positive: np.ndarray = valid[valid > 1e-6]
        if positive.size == 0:
            return np.nan
        return float(positive.size / np.sum(1.0 / positive))

    @cached_property
    def geometric_mean(self) -> float:
        """Geometric mean of valid positive values, or NaN if none exist.

        Only meaningful for features with positive values (e.g., symmetry scores).
        Returns NaN if no valid positive values exist.
        """
        valid: np.ndarray = self.values[self.valid_mask]
        positive: np.ndarray = valid[valid > 1e-6]
        if positive.size == 0:
            return np.nan
        return float(np.exp(np.mean(np.log(positive))))

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
        """Get the value for a specific statistical metric.

        Returns NaN if the statistic is not applicable for this feature type.
        For example, geometric_mean and harmonic_mean are only defined for
        features with positive values. (and not for angles that can be negative).
        """
        if not hasattr(self, statistic.value):
            return np.nan
        return getattr(self, statistic.value)
