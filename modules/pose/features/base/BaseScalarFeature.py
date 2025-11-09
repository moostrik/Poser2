from abc import abstractmethod
from functools import cached_property
from typing import Optional

import numpy as np
from typing_extensions import Self

from modules.pose.features.base.BaseFeature import BaseFeature, FeatureEnum


class BaseScalarFeature(BaseFeature[FeatureEnum]):
    """Base class for scalar features (1D value per joint).

    Use for: angles, distances, normalized values, unbounded scalars, etc.

    Each joint has:
    - A scalar value (float)
    - A confidence score (0.0 to 1.0)

    Values can be NaN to indicate missing/invalid data.

    Subclasses must implement:
    - joint_enum(): Define the joint structure (which IntEnum to use)
    - default_range(): Define valid value range, or (±inf, ±inf) for unbounded
        * construct with (-np.inf, np.inf) for example
    """

    def __init__(self, values: np.ndarray, scores: np.ndarray) -> None:
        """Initialize feature vector with values and scores.

        PERFORMANCE NOTE: No validation performed. Caller must ensure:
        - Both arrays are 1D
        - Arrays match joint_enum() length
        - Scores are in range [0.0, 1.0]

        Use create_validated() for untrusted input or validate() to check.

        Note: Takes ownership of arrays - caller must not modify after passing.

        Args:
            values: Feature values array (1D, float32), length = len(joint_enum())
            scores: Confidence scores (1D, float32, range [0.0, 1.0]), length = len(joint_enum())
        """
        # Get the length from the enum (the source of truth)
        length = len(self.joint_enum())

        # Optional: Keep assertions for development (removed with -O flag)
        assert values.ndim == 1, f"values must be 1D, got {values.ndim}D"
        assert scores.ndim == 1, f"scores must be 1D, got {scores.ndim}D"
        assert len(values) == length, f"values length {len(values)} != enum length {length}"
        assert len(scores) == length, f"scores length {len(scores)} != enum length {length}"

        self._values = values
        self._scores = scores

        # Make immutable
        self._values.flags.writeable = False
        self._scores.flags.writeable = False

        # Compute derived values once
        self._valid_mask = ~np.isnan(self._values)
        self._valid_count = int(np.sum(self._valid_mask))


    # ========== ABSTRACT METHODS ==========

    @classmethod
    @abstractmethod
    def joint_enum(cls) -> type[FeatureEnum]:
        """Get the joint enum type for this class. Must be implemented by subclasses."""
        pass

    @classmethod
    @abstractmethod
    def default_range(cls) -> tuple[float, float]:
        """Define valid range for scalar values. Must be implemented by subclasses."""
        pass

    # ========== PROPERTIES (BaseFeature interface) ==========

    @property
    def scores(self) -> np.ndarray:
        """Confidence scores array (read-only, n_joints)."""
        return self._scores

    @property
    def valid_mask(self) -> np.ndarray:
        """Boolean mask indicating valid (non-NaN) values (n_joints)."""
        return self._valid_mask

    @property
    def valid_count(self) -> int:
        """Number of valid (non-NaN) values."""
        return self._valid_count

    # ========== ADDITIONAL PROPERTIES ==========

    @property
    def values(self) -> np.ndarray:
        """Feature values array (read-only, n_joints)."""
        return self._values

    @property
    def any_valid(self) -> bool:
        """True if at least one valid value is available."""
        return self.valid_count > 0

    @cached_property
    def valid_joints(self) -> list[FeatureEnum]:
        """List of joints with valid values (computed once, cached)."""
        joint_enum_type: type[FeatureEnum] = self.joint_enum()
        return [joint_enum_type(i) for i in np.where(self.valid_mask)[0]]

    # ========== ACCESS ==========

    def __getitem__(self, key: FeatureEnum | int) -> float:
        """Support indexing by joint enum or integer."""
        return float(self.values[key])

    def get(self, joint: FeatureEnum | int, default: float = np.nan) -> float:
        """Get value with default for NaN."""
        value = self._values[joint]
        return float(value) if not np.isnan(value) else default

    def get_score(self, joint: FeatureEnum | int) -> float:
        """Get score for a joint (always between 0.0 and 1.0)."""
        return float(self._scores[joint])

    def to_dict(self, include_invalid: bool = False) -> dict[FeatureEnum, float]:
        """Convert to dictionary mapping joint enums to values."""
        joint_enum_type = self.joint_enum()
        if include_invalid:
            return {joint_enum_type(i): self._values[i] for i in range(len(self._values))}
        else:
            return {joint: self._values[joint] for joint in self.valid_joints}

    # ========== REPRESENTATION ==========

    def __repr__(self) -> str:
        """String representation showing type and validity stats."""
        min_val, max_val = self.default_range()

        min_str = "-inf" if np.isneginf(min_val) else ("inf" if np.isposinf(min_val) else f"{min_val}")
        max_str = "inf" if np.isposinf(max_val) else ("-inf" if np.isneginf(max_val) else f"{max_val}")
        range_str = f", range=({min_str}, {max_str})"

        return f"{self.__class__.__name__}(valid={self.valid_count}/{len(self.values)}{range_str})"

    # ========== CONSTRUCTORS ==========

    @classmethod
    def create_empty(cls) -> Self:
        """Create an empty instance with all NaN values and zero scores."""
        length = len(cls.joint_enum())
        values = np.full(length, np.nan, dtype=np.float32)
        scores = np.zeros(length, dtype=np.float32)
        return cls(values=values, scores=scores)

    @classmethod
    def from_values(cls, values: np.ndarray, scores: Optional[np.ndarray] = None) -> Self:
        """Create instance from values, generating default scores if needed."""
        if scores is None:
            scores = np.where(np.isnan(values), 0.0, 1.0).astype(np.float32)
        return cls(values=values, scores=scores)

    @classmethod
    def create_validated(cls, values: np.ndarray, scores: np.ndarray) -> Self:
        """Create with full validation (use for untrusted input)."""
        instance = cls(values=values, scores=scores)
        is_valid, error = instance.validate(True)
        if not is_valid:
            raise ValueError(f"Invalid {cls.__name__}: {error}")
        return instance

    # ========== VALIDATION (BaseFeature interface) ==========

    def validate(self, check_ranges: bool = True) -> tuple[bool, Optional[str]]:
        """Validate array properties (use for debugging/testing).

        Checks all invariants that should hold for a valid ScalarFeature.
        Returns all validation errors at once for better debugging experience.

        Args:
            check_ranges: Whether to validate score/value ranges (slower)

        Returns:
            (is_valid, error_message): Validation result and error description.
            If invalid, error_message contains all violations separated by "; ".
        """
        errors = []

        # Check dimensions
        if self._values.ndim != 1:
            errors.append(f"values must be 1D, got shape {self._values.shape}")
        if self._scores.ndim != 1:
            errors.append(f"scores must be 1D, got shape {self._scores.shape}")

        # Early return for structural errors (can't continue validation)
        if errors:
            return (False, "; ".join(errors))

        # Check enum length match
        length = len(self.joint_enum())
        if len(self._values) != length:
            errors.append(f"values length {len(self._values)} != {self.joint_enum().__name__} length {length}")
        if len(self._scores) != length:
            errors.append(f"scores length {len(self._scores)} != {self.joint_enum().__name__} length {length}")

        # Check arrays match each other
        if len(self._values) != len(self._scores):
            errors.append(f"Length mismatch: values={len(self._values)} vs scores={len(self._scores)}")

        # Early return for length errors (can't continue validation)
        if errors:
            return (False, "; ".join(errors))

        # Check NaN/score consistency
        invalid_mask = ~self._valid_mask
        invalid_indices = np.where(invalid_mask)[0]
        if len(invalid_indices) > 0:
            bad_scores = self._scores[invalid_mask] != 0.0
            if np.any(bad_scores):
                bad_indices = invalid_indices[bad_scores]
                joint_enum = self.joint_enum()
                bad_joints = [joint_enum(i).name for i in bad_indices[:3]]
                more = f" and {len(bad_indices) - 3} more" if len(bad_indices) > 3 else ""
                errors.append(f"NaN values must have score 0.0, but found non-zero scores at: {', '.join(bad_joints)}{more}")

        # Range checks (expensive, only if requested)
        if check_ranges and self._valid_count > 0:
            # Validate score range
            valid_scores = self._scores[self._valid_mask]
            if np.any((valid_scores < 0.0) | (valid_scores > 1.0)):
                errors.append(f"Scores outside [0.0, 1.0]: min={valid_scores.min():.3f}, max={valid_scores.max():.3f}")

            # Validate value range
            min_val, max_val = self.default_range()
            valid_values = self._values[self._valid_mask]

            # Check lower bound (only if not -inf)
            if not np.isneginf(min_val):
                below_min = valid_values < min_val
                if np.any(below_min):
                    errors.append(f"values below minimum {min_val}: min={valid_values[below_min].min():.2f}")

            # Check upper bound (only if not +inf)
            if not np.isposinf(max_val):
                above_max = valid_values > max_val
                if np.any(above_max):
                    errors.append(f"values above maximum {max_val}: max={valid_values[above_max].max():.2f}")

        # Return result
        if errors:
            return (False, "; ".join(errors))
        return (True, None)

