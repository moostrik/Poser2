from abc import ABC, abstractmethod
from enum import IntEnum
from functools import cached_property
from typing import Generic, TypeVar, Optional

# Third-party imports
import numpy as np
from typing_extensions import Self


PoseEnum = TypeVar('PoseEnum', bound=IntEnum)


class PoseVectorFeatureBase(ABC, Generic[PoseEnum]):
    """Base class for all pose feature vectors.

    Provides common functionality for storing values with associated confidence scores,
    handling invalid (NaN) data, indexing operations, and joint enum mapping.

    Design principles:
    - Immutable after creation
    - Always returns valid objects (uses NaN for missing data)
    - Consistent indexing and slicing behavior
    - Validity determined by NaN presence, not None checks
    - Joint-based indexing via enum
    - Optional range validation for values

    Note: Base class assumes 1D values array. PosePoints overrides for 2D.
    """

    def __init__(self, values: np.ndarray, scores: Optional[np.ndarray] = None) -> None:
        """Initialize feature vector with values and optional scores.

        Note: Takes ownership of arrays - caller should not modify them after passing.

        Args:
            values: Feature values array (1D)
            scores: Confidence scores per feature (1D). If None, defaults to 1.0 for valid, 0.0 for NaN
        """
        self._values = values
        self._scores = self._initialize_scores(scores)

        # Make immutable
        self._values.flags.writeable = False
        self._scores.flags.writeable = False

        self._validate()

    def _initialize_scores(self, scores: Optional[np.ndarray]) -> np.ndarray:
        """Initialize scores array from input or generate default scores."""
        if scores is None:
            # Generate default scores (must create new array)
            return np.where(np.isnan(self._values), 0.0, 1.0).astype(np.float32)
        else:
            # Take ownership - no copy needed
            return scores

    def _validate(self) -> None:
        """Validate shape and data constraints."""
        # Validate values are 1D
        if self._values.ndim != 1:
            raise ValueError(f"values array must be 1D, got shape {self._values.shape}")

        # Validate scores are 1D
        if self._scores.ndim != 1:
            raise ValueError(f"scores array must be 1D, got shape {self._scores.shape}")

        # Validate array length matches joint enum
        joint_enum_type = self.joint_enum()
        expected_length = len(joint_enum_type)

        actual_length = len(self._values)
        if actual_length != expected_length:
            raise ValueError(
                f"values array length ({actual_length}) must match joint enum length ({expected_length})"
            )
        if len(self._scores) != expected_length:
            raise ValueError(
                f"scores array length ({len(self._scores)}) must match joint enum length ({expected_length})"
            )

        # Validate: NaN values must have 0.0 scores
        nan_mask = np.isnan(self._values)
        invalid = nan_mask & (self._scores > 0.0)

        if np.any(invalid):
            invalid_joints = [joint_enum_type(i).name for i in np.where(invalid)[0]]
            raise ValueError(
                f"Data integrity violation: NaN values must have 0.0 scores. "
                f"Invalid joints: {', '.join(invalid_joints)}"
            )

        # Validate value range (if specified)
        value_range = self.default_range()
        if value_range is not None:
            min_val, max_val = value_range
            valid_values = self._values[~nan_mask]

            if valid_values.size > 0:
                out_of_range = (valid_values < min_val) | (valid_values > max_val)
                if np.any(out_of_range):
                    violating_values = valid_values[out_of_range]
                    raise ValueError(
                        f"Values must be in range [{min_val}, {max_val}]. "
                        f"Found {np.sum(out_of_range)} values out of range. "
                        f"Examples: {violating_values[:3]}"
                    )

    # ========== ABSTRACT METHODS ==========

    @classmethod
    @abstractmethod
    def joint_enum(cls) -> type[PoseEnum]:
        """Get the joint enum type for this class. Must be implemented by subclasses."""
        pass

    @classmethod
    def default_range(cls) -> Optional[tuple[float, float]]:
        """Override in subclasses to define valid range."""
        return None

    # ========== PROPERTIES ==========

    @property
    def values(self) -> np.ndarray:
        """Feature values array (read-only)."""
        return self._values

    @property
    def scores(self) -> np.ndarray:
        """Confidence scores array (read-only)."""
        return self._scores

    @cached_property
    def valid_mask(self) -> np.ndarray:
        """Boolean mask indicating valid (non-NaN) features."""
        return ~np.isnan(self._values)

    @cached_property
    def valid_count(self) -> int:
        """Number of valid (non-NaN) features."""
        return int(np.sum(self.valid_mask))

    @cached_property
    def any_valid(self) -> bool:
        """True if at least one valid value is available."""
        return self.valid_count > 0

    @cached_property
    def valid_joints(self) -> list[PoseEnum]:
        """List of joints with valid (non-zero score) values."""
        joint_enum_type = self.joint_enum()
        return [joint_enum_type(i) for i in np.where(self.valid_mask)[0]]

    # ========== ACCESS ==========

    def __len__(self) -> int:
        """Return total number of features."""
        return len(self.values)

    def __getitem__(self, key: PoseEnum | int) -> float:
        """Support indexing by joint enum or integer."""
        return float(self.values[key])

    def get(self, joint: PoseEnum | int, default: float = np.nan) -> float:
        """Get value with default for NaN."""
        value = self._values[joint]
        return float(value) if not np.isnan(value) else default

    def get_score(self, joint: PoseEnum | int) -> float:
        """Get score for a joint (always between 0.0 and 1.0)."""
        return float(self._scores[joint])

    def to_dict(self, include_invalid: bool = False) -> dict[PoseEnum, float]:
        """Convert to dictionary mapping joint names to values."""
        joint_enum_type = self.joint_enum()
        if include_invalid:
            return {joint_enum_type(i): self._values[i] for i in range(len(self._values))}
        else:
            return {joint: self._values[joint] for joint in self.valid_joints}

    def __repr__(self) -> str:
        """String representation showing type and validity stats."""
        range_str = f", range={self.default_range()}" if self.default_range() else ""
        return f"{self.__class__.__name__}(valid={self.valid_count}/{len(self.values)}{range_str})"

    # ========== CONSTRUCTORS ==========

    @classmethod
    def create_empty(cls) -> Self:
        """Create an empty instance with all NaN values."""
        joint_enum_type = cls.joint_enum()
        values = np.full(len(joint_enum_type), np.nan, dtype=np.float32)
        scores = np.zeros(len(joint_enum_type), dtype=np.float32)
        return cls(values=values, scores=scores)

