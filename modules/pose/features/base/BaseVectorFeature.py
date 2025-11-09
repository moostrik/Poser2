from abc import abstractmethod
from functools import cached_property
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
    - default_range(): Define valid value range per dimension
        * Returns array of shape (n_dims, 2) with [[min, max], ...] for each dimension
        * Use np.inf for unbounded dimensions: np.array([[-np.inf, np.inf], ...])
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
    def default_range(cls) -> np.ndarray:
        """Define valid value ranges per dimension. Must be implemented by subclasses.

        Note:
            - Shape must be (dimensions(), 2)
            - Each row is [min, max] for that dimension
            - Use np.inf for unbounded dimensions
            - Infinite ranges skip validation checks (automatic optimization)
        """
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
        """Vector values array (read-only, shape: n_joints × n_dims)."""
        return self._values

    @property
    def any_valid(self) -> bool:
        """True if at least one valid vector is available."""
        return self.valid_count > 0

    @cached_property
    def valid_joints(self) -> list[FeatureEnum]:
        """List of joints with valid values (computed once, cached)."""
        joint_enum_type: type[FeatureEnum] = self.joint_enum()
        return [joint_enum_type(i) for i in np.where(self.valid_mask)[0]]

    # ========== ACCESS ==========

    def __getitem__(self, key: FeatureEnum | int) -> np.ndarray:
        """Get vector for a joint."""
        return self._values[key]

    def get_component(self, joint: FeatureEnum | int, dim: int) -> float:
        """Get specific component of a vector.

        Args:
            joint: Joint enum member or index
            dim: Dimension index (0 for first, 1 for second, etc.)

        Returns:
            Component value (may be NaN)
        """
        return float(self._values[joint, dim])

    def get(self, joint: FeatureEnum | int, default: float = np.nan) -> np.ndarray:
        """Get vector with NaN components replaced by default value.

        Args:
            joint: Joint enum member or index
            default: Value to replace NaN components with. If np.nan (default),
                     returns vector unchanged (including NaN components).

        Returns:
            Vector with NaN components optionally replaced.
            If default is np.nan or vector has no NaNs, returns original (no copy).
            Otherwise returns a copy with NaNs replaced.

        Examples:
            >>> # Get vector as-is (including NaNs)
            >>> vector = feature.get(BodyJoint.SHOULDER)

            >>> # Replace NaNs with 0.0
            >>> vector = feature.get(BodyJoint.SHOULDER, default=0.0)
            >>> # [100.5, NaN, 200.3] → [100.5, 0.0, 200.3]
        """
        vector = self._values[joint]

        # Fast path: return original if no replacement needed
        if np.isnan(default) or not np.any(np.isnan(vector)):
            return vector

        # Slow path: copy and replace NaNs
        vector = vector.copy()
        nan_mask = np.isnan(vector)
        vector[nan_mask] = default
        return vector

    def get_score(self, joint: FeatureEnum | int) -> float:
        """Get score for a joint (always between 0.0 and 1.0)."""
        return float(self._scores[joint])

    def to_dict(self, include_invalid: bool = False) -> dict[FeatureEnum, np.ndarray]:
        """Convert to dictionary mapping joint enums to values.

        Args:
            include_invalid: If True, include values with NaN components

        Returns:
            Dictionary of joint → vector array mappings
        """
        joint_enum_type = self.joint_enum()
        if include_invalid:
            return {joint_enum_type(i): self._values[i] for i in range(len(self._values))}
        else:
            return {joint: self._values[joint] for joint in self.valid_joints}

    # ========== REPRESENTATION ==========

    def __repr__(self) -> str:
        """String representation showing type, validity stats, and dimensions."""
        return f"{self.__class__.__name__}(valid={self.valid_count}/{len(self._values)}, dims={self.dimensions()})"

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
        """Create with full validation (use for untrusted input).

        Performs complete validation including:
        - Structural checks (dimensions, lengths)
        - NaN/score consistency
        - Score range validation
        - Component range validation per dimension

        Args:
            values: Vector values array
            scores: Confidence scores array

        Returns:
            New validated feature instance

        Raises:
            ValueError: If any validation check fails

        Note: For performance-critical code, use cls(values, scores) directly
              and validate explicitly if needed.
        """
        instance = cls(values=values, scores=scores)
        is_valid, error = instance.validate(check_ranges=True)
        if not is_valid:
            raise ValueError(f"Invalid {cls.__name__}: {error}")
        return instance

    # ========== VALIDATION (BaseFeature interface) ==========

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
        - Components are within default_range() per dimension (if check_ranges=True)
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

            # Validate component ranges per dimension
            range_limits = self.default_range()
            valid_values = self._values[self._valid_mask]  # Shape: (n_valid, n_dims)

            dim_names = ['x', 'y', 'z', 'w']
            for dim in range(n_dims):
                min_val, max_val = range_limits[dim]
                dim_values = valid_values[:, dim]
                dim_name = dim_names[dim] if dim < len(dim_names) else f"dim{dim}"

                # Check lower bound (only if not -inf)
                if not np.isneginf(min_val):
                    below_min = dim_values < min_val
                    if np.any(below_min):
                        errors.append(f"{dim_name} below minimum {min_val}: min={dim_values[below_min].min():.2f}")

                # Check upper bound (only if not +inf)
                if not np.isposinf(max_val):
                    above_max = dim_values > max_val
                    if np.any(above_max):
                        errors.append(f"{dim_name} above maximum {max_val}: max={dim_values[above_max].max():.2f}")

        # Return result
        if errors:
            return (False, "; ".join(errors))
        return (True, None)