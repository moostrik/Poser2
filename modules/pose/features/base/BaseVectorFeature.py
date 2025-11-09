from abc import abstractmethod
from typing import Optional

import numpy as np
from typing_extensions import Self

from modules.pose.features.base.BaseFeature import BaseFeature, FeatureEnum


class BaseVectorFeature(BaseFeature[FeatureEnum]):
    """Base class for vector features (multi-dimensional values per joint).

    Use for: 2D/3D keypoints, direction vectors, velocity vectors, etc.

    Each joint has:
    - A vector with n dimensions (e.g., x, y for 2D points)
    - A confidence score (0.0 to 1.0)

    Vectors can have NaN components to indicate missing/invalid data.
    A vector is considered invalid if ANY of its components is NaN.

    Subclasses must implement:
    - joint_enum(): Define the joint structure (which IntEnum to use)
    - dimensions(): Number of dimensions per vector (2 for 2D, 3 for 3D)
    - default_range(): Define valid value range for ALL dimensions
    """

    def __init__(self, values: np.ndarray, scores: np.ndarray) -> None:
        """Initialize vector feature with multi-dimensional values and scores.

        PERFORMANCE NOTE: No validation performed. Caller must ensure:
        - values is 2D with shape (n_joints, n_dims)
        - scores is 1D with length n_joints
        - Dimensions match dimensions()
        - Scores are in range [0.0, 1.0]

        Use create_validated() for untrusted input or validate() to check.

        Note: Takes ownership of arrays - caller must not modify after passing.
        """
        # Get expected dimensions from the enum and subclass
        length = len(self.joint_enum())
        n_dims = self.dimensions()

        # Optional: Keep assertions for development (removed with -O flag)
        assert values.ndim == 2, f"values must be 2D, got {values.ndim}D"
        assert values.shape == (length, n_dims), \
            f"values shape {values.shape} != expected ({length}, {n_dims})"
        assert scores.ndim == 1, f"scores must be 1D, got {scores.ndim}D"
        assert len(scores) == length, f"scores length {len(scores)} != enum length {length}"

        self._values = values
        self._scores = scores

        # Make immutable
        self._values.flags.writeable = False
        self._scores.flags.writeable = False

        # Compute derived values once
        # A vector is valid only if ALL its components are non-NaN
        self._valid_mask = ~np.any(np.isnan(self._values), axis=1)
        self._valid_count = int(np.sum(self._valid_mask))

    # ========== ABSTRACT METHODS ==========

    @classmethod
    @abstractmethod
    def joint_enum(cls) -> type[FeatureEnum]:
        """Get the joint enum type for this class. Must be implemented by subclasses."""
        pass

    @classmethod
    @abstractmethod
    def dimensions(cls) -> int:
        """Number of dimensions per vector (2 for 2D, 3 for 3D, etc.). Must be implemented by subclasses."""
        pass

    @classmethod
    @abstractmethod
    def default_range(cls) -> tuple[float, float]:
        """Define valid value range. Must be implemented by subclasses."""
        pass

    # ========== BASEFEATURE ==========

    def __len__(self) -> int:
        """Number of joints in this feature."""
        return len(self.joint_enum())

    # ========== RAW DATA ACCESS ==========

    def __getitem__(self, key: FeatureEnum | int) -> np.ndarray:
        """Get vector for a joint."""
        return self._values[key]

    @property
    def values(self) -> np.ndarray:
        """Vector values array (read-only, shape: n_joints × n_dims)."""
        return self._values

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

    # ========== FRIENDLY ACCESS ==========

    def get_score(self, joint: FeatureEnum | int) -> float:
        """Get confidence score for a single joint (Python float)."""
        return float(self._scores[joint])

    def get_scores(self, joints: list[FeatureEnum | int]) -> list[float]:
        """Get confidence scores for multiple joints (Python list)."""
        return [float(self._scores[joint]) for joint in joints]

    def get_valid(self, joint: FeatureEnum | int) -> bool:
        """Check if the value for a joint is valid (not NaN)."""
        return self._valid_mask[joint]

    def are_valid(self, joints: list[FeatureEnum | int]) -> bool:
        """Check if ALL specified joints are valid (batch validation)."""
        return bool(np.all(self._valid_mask[list(joints)]))

    # ========== REPRESENTATION ==========

    def __repr__(self) -> str:
        """String representation showing type, validity stats, and dimensions."""
        return f"{self.__class__.__name__}(valid={self.valid_count}/{len(self)}, dims={self.dimensions()})"

    # ========== CONSTRUCTORS ==========

    @classmethod
    def create_empty(cls) -> Self:
        """Create an empty instance with all NaN values and zero scores."""
        length = len(cls.joint_enum())
        n_dims = cls.dimensions()
        values = np.full((length, n_dims), np.nan, dtype=np.float32)
        scores = np.zeros(length, dtype=np.float32)
        return cls(values=values, scores=scores)

    @classmethod
    def from_values(cls, values: np.ndarray, scores: Optional[np.ndarray] = None) -> Self:
        """Create instance from values, generating default scores if needed.

        Args:
            values: Value array (shape: n_joints × n_dims)
            scores: Optional scores array. If None, generates scores:
                    - 1.0 for valid values (all components non-NaN)
                    - 0.0 for invalid values (any component is NaN)

        Returns:
            New feature instance
        """
        if scores is None:
            # Valid if ALL components are non-NaN
            has_nan = np.any(np.isnan(values), axis=1)
            scores = np.where(has_nan, 0.0, 1.0).astype(np.float32)
        return cls(values=values, scores=scores)

    @classmethod
    def create_validated(cls, values: np.ndarray, scores: np.ndarray) -> Self:
        """Create with full validation (use for untrusted input)."""
        instance = cls(values=values, scores=scores)
        is_valid, error = instance.validate(check_ranges=True)
        if not is_valid:
            raise ValueError(f"Invalid {cls.__name__}: {error}")
        return instance

    # ========= VALIDATION ==========

    def validate(self, check_ranges: bool = True) -> tuple[bool, Optional[str]]:
        """Validate array properties (use for debugging/testing).

        Checks all invariants that should hold for a valid VectorFeature.
        Returns all validation errors at once for better debugging experience.

        Checks:
        - values array is 2D
        - values shape matches (n_joints, n_dims)
        - scores array is 1D
        - scores length matches n_joints
        - Vectors with any NaN component must have score 0.0
        - Scores are in [0.0, 1.0] (if check_ranges=True)
        - ALL components are within default_range() (if check_ranges=True)
            * Same range applies to all dimensions (x, y, z, ...)
            * Infinite bounds (±inf) are automatically skipped during validation

        Args:
            check_ranges: Whether to validate score/component ranges (slower)

        Returns:
            (is_valid, error_message): Validation result and error description.
            If invalid, error_message contains all violations separated by "; ".
        """
        errors = []

        # Check dimensions
        if self._values.ndim != 2:
            errors.append(f"values must be 2D, got shape {self._values.shape}")
        if self._scores.ndim != 1:
            errors.append(f"scores must be 1D, got shape {self._scores.shape}")

        # Early return for structural errors (can't continue validation)
        if errors:
            return (False, "; ".join(errors))

        # Check shape match
        length = len(self.joint_enum())
        n_dims = self.dimensions()
        expected_shape = (length, n_dims)

        if self._values.shape != expected_shape:
            errors.append(f"values shape {self._values.shape} != expected {expected_shape}")

        if len(self._scores) != length:
            errors.append(f"scores length {len(self._scores)} != {self.joint_enum().__name__} length {length}")

        # Early return for length errors (can't continue validation)
        if errors:
            return (False, "; ".join(errors))

        # Check NaN/score consistency: Vectors with ANY NaN component MUST have score 0.0
        invalid_mask = ~self._valid_mask
        invalid_indices = np.where(invalid_mask)[0]
        if len(invalid_indices) > 0:
            bad_scores = self._scores[invalid_mask] != 0.0
            if np.any(bad_scores):
                bad_indices = invalid_indices[bad_scores]
                joint_enum = self.joint_enum()
                bad_joints = [joint_enum(i).name for i in bad_indices[:3]]
                more = f" and {len(bad_indices) - 3} more" if len(bad_indices) > 3 else ""
                errors.append(f"Invalid values must have score 0.0, but found non-zero scores at: {', '.join(bad_joints)}{more}")

        # Range checks (expensive, only if requested)
        if check_ranges and self._valid_count > 0:
            # Validate score range
            valid_scores = self._scores[self._valid_mask]
            if np.any((valid_scores < 0.0) | (valid_scores > 1.0)):
                errors.append(f"Scores outside [0.0, 1.0]: min={valid_scores.min():.3f}, max={valid_scores.max():.3f}")

            # Validate component ranges (same range for ALL dimensions)
            min_val, max_val = self.default_range()  # Single tuple applies to all dimensions
            valid_values = self._values[self._valid_mask]  # Shape: (n_valid, n_dims), all components

            # Check ALL components against the same range
            # Check lower bound (only if not -inf)
            if not np.isneginf(min_val):
                below_min = valid_values < min_val
                if np.any(below_min):
                    errors.append(f"Components below minimum {min_val}: min={valid_values[below_min].min():.2f}")

            # Check upper bound (only if not +inf)
            if not np.isposinf(max_val):
                above_max = valid_values > max_val
                if np.any(above_max):
                    errors.append(f"Components above maximum {max_val}: max={valid_values[above_max].max():.2f}")

        # Return result
        if errors:
            return (False, "; ".join(errors))
        return (True, None)