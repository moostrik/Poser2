"""
=============================================================================
BASESCALARFEATURE API REFERENCE
=============================================================================

Base class for scalar features (single value per element).

Use for: angles, distances, normalized values, unbounded scalars, etc.

Summary of BaseFeature Design Philosophy:
==========================================

Immutability & Ownership:
  • Features are IMMUTABLE - arrays set to read-only after construction
  • Constructor takes OWNERSHIP - caller must not modify arrays after passing
  • Modifications create new features (functional style)

Data Access (two patterns):
  Raw (numpy):     feature.values, feature.scores, feature[element]
  Python-friendly: feature.get(element, fill), feature.get_score(element)

NaN Semantics:
  • Invalid data = NaN with score 0.0 (enforced)
  • Use get(element, fill=0.0) for automatic NaN handling

Cached Properties:
  • Subclasses may add @cached_property (safe due to immutability)

Construction:
  • MyFeature(values, scores)           → Direct (minimal assertions)
  • MyFeature.create_empty()            → All NaN values, zero scores

Validation:
  • Minimal assertions in constructors (removed with -O flag for production)
  • validate() method for debugging/testing/untrusted input
  • Fast by default, validate only when needed

Performance:
  Fast:     Property access, indexing, cached properties, array ops
  Moderate: get(), get_score() (Python conversion)
  Slow:     get_values(), get_scores() (iteration), validate()

BaseScalarFeature-Specific:
============================

Structure:
----------
Each element has:
  • A scalar value (float) - may be NaN for invalid/missing data
  • A confidence score [0.0, 1.0]

Storage:
  • values: np.ndarray, shape (n_elements,), dtype float32
  • scores: np.ndarray, shape (n_elements,), dtype float32

Properties:
-----------
  • values: np.ndarray                             All scalar values (n_elements,)
  • scores: np.ndarray                             All confidence scores (n_elements,)
  • valid_mask: np.ndarray                         Boolean validity mask (n_elements,)
  • valid_count: int                               Number of valid values
  • len(feature): int                              Total number of elements

Single Value Access:
--------------------
  • feature[element] -> float                      Get value (supports enum or int)
  • feature.get(element, fill=np.nan) -> float     Get value with NaN handling
  • feature.get_value(element, fill) -> float      Alias for get()
  • feature.get_score(element) -> float            Get confidence score
  • feature.get_valid(element) -> bool             Check if value is valid

Batch Operations:
-----------------
  • feature.get_values(elements, fill) -> list[float]  Get multiple values
  • feature.get_scores(elements) -> list[float]        Get multiple scores
  • feature.are_valid(elements) -> bool                Check if ALL valid

Factory Methods:
----------------
  • MyFeature.create_empty() -> MyFeature          All NaN values, zero scores

Validation:
-----------
  • feature.validate(check_ranges=True) -> tuple[bool, str|None]
      Returns (is_valid, error_message)

Abstract Methods (must implement in subclasses):
-------------------------------------------------
  • enum() -> type[IntEnum]                Feature element enum
  • range() -> tuple[float, float]         Valid value range
      Optional convenience constants: NORMALIZED_RANGE, POSITIVE_RANGE, UNBOUNDED_RANGE,
                                      PI_RANGE, TWO_PI_RANGE, SYMMETRIC_PI_RANGE

Notes:
------
- All values are single floats (scalar per element)
- Invalid values are NaN with score 0.0
- Arrays are read-only after construction (immutable)
- Use validate() for debugging, not in production loops
- Constructor takes ownership - caller must not modify arrays after passing
=============================================================================
"""
from abc import abstractmethod
from typing import Optional

import numpy as np
from typing_extensions import Self

from modules.pose.features.base.BaseFeature import BaseFeature, FeatureEnum


class BaseScalarFeature(BaseFeature[FeatureEnum]):
    """Base class for scalar features (1D value per element).

    See module docstring for full API reference.
    Inherits design philosophy from BaseFeature.
    """

    def __init__(self, values: np.ndarray, scores: np.ndarray) -> None:
        """Initialize scalar feature with values and scores.

        PERFORMANCE NOTE: No validation performed. Caller must ensure:
        - Both arrays are 1D
        - Arrays match length()
        - Scores are in range [0.0, 1.0]

        Use create_validated() for untrusted input or validate() to check.

        Note: Takes ownership of arrays - caller must not modify after passing.

        Args:
            values: Feature values array (1D, float32), length = length()
            scores: Confidence scores (1D, float32, range [0.0, 1.0]), length = length()
        """
        # Use length() as source of truth (works for both enum and runtime-configured)
        length = self.length()

        # Optional: Keep assertions for development (removed with -O flag)
        assert values.ndim == 1, f"values must be 1D, got {values.ndim}D"
        assert scores.ndim == 1, f"scores must be 1D, got {scores.ndim}D"
        assert len(values) == length, f"values length {len(values)} != expected length {length}"
        assert len(scores) == length, f"scores length {len(scores)} != expected length {length}"

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
    def enum(cls) -> type[FeatureEnum]:
        """Feature enum defining the elements of this feature."""
        pass

    @classmethod
    def length(cls) -> int:
        """Number of elements. Override to return len(enum()) or configured value."""
        return len(cls.enum())

    @classmethod
    @abstractmethod
    def range(cls) -> tuple[float, float]:
        """Define valid range for scalar values. Must be implemented by subclasses."""
        pass

    @classmethod
    def display_range(cls) -> tuple[float, float]:
        """Display range for visualization. Defaults to range().

        Override in subclasses where the validation range is too wide
        or unbounded for meaningful display.
        """
        return cls.range()

    # ========== BASEFEATURE ==========

    def __len__(self) -> int:
        """Number of elements in this feature."""
        return self.length()

    # ========== RAW DATA ACCESS ==========

    def __getitem__(self, element: FeatureEnum | int) -> float:
        """Support indexing by feature enum or integer."""
        return float(self.values[element])

    @property
    def values(self) -> np.ndarray:
        """Feature values array (read-only, n_elements)."""
        return self._values

    @property
    def scores(self) -> np.ndarray:
        """Confidence scores array (read-only, n_elements)."""
        return self._scores

    @property
    def valid_mask(self) -> np.ndarray:
        """Boolean mask indicating valid (non-NaN) values (n_elements)."""
        return self._valid_mask

    @property
    def valid_count(self) -> int:
        """Number of valid (non-NaN) values."""
        return self._valid_count

    # ========== ACCESS ==========

    def get(self, element: FeatureEnum | int, fill: float = np.nan) -> float:
        """Get raw value for an element (may be NaN), with fill for NaN."""
        value = self._values[element]
        return float(value) if not np.isnan(value) else fill

    def get_value(self, element: FeatureEnum | int, fill: float = np.nan) -> float:
        """Alias for get() to emphasize value retrieval."""
        return self.get(element, fill)

    def get_values(self, elements: list[FeatureEnum | int], fill: float = np.nan) -> list[float]:
        """Get values for multiple elements with fill for NaN."""
        values = self._values[list(elements)]

        # Fast path: no NaN replacement needed
        if np.isnan(fill):
            return values.tolist()

        # Vectorized NaN replacement
        valid_mask = self._valid_mask[list(elements)]
        result = np.where(valid_mask, values, fill)
        return result.tolist()

    def get_score(self, element: FeatureEnum | int) -> float:
        """Get score for an element (always between 0.0 and 1.0)."""
        return float(self._scores[element])

    def get_scores(self, elements: list[FeatureEnum | int]) -> list[float]:
        """Get confidence scores for multiple elements."""
        return [float(self._scores[element]) for element in elements]

    def get_valid(self, element: FeatureEnum | int, validate: bool = False) -> bool:
        """Check if the value for an element is valid (not NaN).

        Args:
            element: Element to check
            validate: If True, performs extra validation checks on this element

        Returns:
            True if the element has a valid (non-NaN) value
        """
        is_valid = bool(self._valid_mask[element])

        if validate:
            # Check if mask matches actual data
            actual_valid = not np.isnan(self._values[element])
            if is_valid != actual_valid:
                print(f"VALIDATION ERROR: {self.enum()(element).name} mask={is_valid} but actual={actual_valid}")
                print(f"  Value: {self._values[element]}")
                print(f"  Score: {self._scores[element]}")

            # Check if invalid values have score 0.0
            if not is_valid and self._scores[element] != 0.0:
                print(f"VALIDATION ERROR: {self.enum()(element).name} is invalid but has non-zero score {self._scores[element]}")

        return is_valid

    def are_valid(self, elements: list[FeatureEnum | int]) -> bool:
        """Check if ALL specified elements are valid (batch validation)."""
        return bool(np.all(self._valid_mask[list(elements)]))

    # ========== REPRESENTATION ==========

    def __repr__(self) -> str:
        """String representation showing type and validity stats."""
        return f"{self.__class__.__name__}(valid={self.valid_count}/{len(self)})"

    # ========== FACTORY METHODS ==========

    _empty_instance: Optional[Self] = None  # Class-level cache for the empty instance

    @classmethod
    def create_dummy(cls) -> Self:
        """Create empty feature with all NaN values and zero scores.

        Uses a class-level cache to ensure the empty instance is created only once per class.
        """
        if cls._empty_instance is None:
            values = np.full(cls.length(), np.nan, dtype=np.float32)
            scores = np.zeros(cls.length(), dtype=np.float32)
            cls._empty_instance = cls(values=values, scores=scores)
        return cls._empty_instance

    @classmethod # KEEP FOR NOW FOR BACKWARDS COMPATIBILITY
    def from_values(cls, values: np.ndarray, scores: Optional[np.ndarray] = None) -> Self:
        """Create instance from values, generating default scores if needed."""
        if scores is None:
            scores = np.where(np.isnan(values), 0.0, 1.0).astype(np.float32)
        return cls(values=values, scores=scores)

    # ========= VALIDATION ==========

    def validate(self, check_ranges: bool = True) -> tuple[bool, Optional[str]]:
        """Validate array properties (use for debugging/testing).

        Checks all invariants that should hold for a valid ScalarFeature.
        Returns all validation errors at once for better debugging experience.

        Checks:
        - values array is 1D
        - values length matches n_elements
        - scores array is 1D
        - scores length matches n_elements
        - valid_mask matches actual NaN state (cached vs computed)
        - valid_count matches actual count
        - NaN values must have score 0.0
        - Scores are in [0.0, 1.0] (if check_ranges=True)
        - Values are within range() (if check_ranges=True)
            * Infinite bounds (±inf) are automatically skipped during validation

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

        # Check shape match - use length() as source of truth
        length = self.length()

        if len(self._values) != length:
            errors.append(f"values length {len(self._values)} != expected length {length}")

        if len(self._scores) != length:
            errors.append(f"scores length {len(self._scores)} != expected length {length}")

        # Early return for length errors (can't continue validation)
        if errors:
            return (False, "; ".join(errors))

        # Check valid_mask consistency (cached vs actual)
        actual_valid_mask = ~np.isnan(self._values)
        if not np.array_equal(self._valid_mask, actual_valid_mask):
            mismatches = np.where(self._valid_mask != actual_valid_mask)[0]
            feature_enum = self.enum()
            mismatch_elements = [feature_enum(i).name for i in mismatches[:3]]
            more = f" and {len(mismatches) - 3} more" if len(mismatches) > 3 else ""
            errors.append(f"Cached valid_mask doesn't match actual NaN state at: {', '.join(mismatch_elements)}{more}")

        # Check valid_count consistency
        actual_valid_count = int(np.sum(actual_valid_mask))
        if self._valid_count != actual_valid_count:
            errors.append(f"Cached valid_count {self._valid_count} != actual {actual_valid_count}")

        # Check NaN/score consistency: NaN values MUST have score 0.0
        invalid_mask = ~self._valid_mask
        invalid_indices = np.where(invalid_mask)[0]
        if len(invalid_indices) > 0:
            bad_scores = self._scores[invalid_mask] != 0.0
            if np.any(bad_scores):
                bad_indices = invalid_indices[bad_scores]
                feature_enum = self.enum()
                bad_elements = [feature_enum(i).name for i in bad_indices[:3]]
                more = f" and {len(bad_indices) - 3} more" if len(bad_indices) > 3 else ""
                errors.append(f"NaN values must have score 0.0, but found non-zero scores at: {', '.join(bad_elements)}{more}")

        # Range checks (expensive, only if requested)
        if check_ranges and self._valid_count > 0:
            # Validate score range
            valid_scores = self._scores[self._valid_mask]
            if np.any((valid_scores < 0.0) | (valid_scores > 1.0)):
                errors.append(f"Scores outside [0.0, 1.0]: min={valid_scores.min():.3f}, max={valid_scores.max():.3f}")

            # Validate value range
            min_val, max_val = self.range()
            valid_values = self._values[self._valid_mask]

            # Check lower bound (only if not -inf)
            if not np.isneginf(min_val):
                below_min = valid_values < min_val
                if np.any(below_min):
                    errors.append(f"Values below minimum {min_val}: min={valid_values[below_min].min():.2f}")

            # Check upper bound (only if not +inf)
            if not np.isposinf(max_val):
                above_max = valid_values > max_val
                if np.any(above_max):
                    errors.append(f"Values above maximum {max_val}: max={valid_values[above_max].max():.2f}")

        # Return result
        if errors:
            return (False, "; ".join(errors))
        return (True, None)
