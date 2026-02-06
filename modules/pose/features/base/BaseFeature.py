"""
=============================================================================
BASEFEATURE API REFERENCE
=============================================================================

Base interface for all pose features. Defines common structure, data access
patterns, and design philosophy shared across all feature types.

Design Philosophy:
==================

Immutability & Ownership:
-------------------------
• Features are IMMUTABLE after construction
• Arrays are set to read-only (writeable=False)
• Constructor takes OWNERSHIP of arrays - caller must not modify
• Create new features for modifications (functional style)
• Enables safe caching of computed properties

Raw Access (numpy-native):
--------------------------
• feature.values      → Full array (shape varies by type), read-only
• feature.scores      → Full scores (n_elements,), read-only
• feature[element]    → Single value (type varies by subclass)

Use for: Numpy operations, batch processing, performance-critical code
Performance: O(1) array access, no Python type conversion

Python-Friendly Access:
-----------------------
• feature.get(element, fill)     → Python types with NaN handling
• feature.get_score(element)     → Python float [0.0, 1.0]
• feature.get_scores(elements)   → Python list

Use for: Logic, conditionals, unpacking, defaults, user-facing code
Performance: Slightly slower due to type conversion

NaN Semantics:
--------------
• Invalid/missing data is represented as NaN (np.nan)
• NaN values MUST have score 0.0 (enforced by validation)
• Use valid_mask to check validity before accessing
• Use get(element, fill=0.0) to handle NaN automatically

Cached Properties:
------------------
• Subclasses may provide cached computed properties
• Safe because features are immutable
• Use @cached_property from functools
• First access computes and caches, subsequent O(1) lookup

Construction Patterns:
----------------------
• MyFeature(values, scores)              → Direct (minimal assertions)
• MyFeature.create_empty()               → All NaN values, zero scores

Validation Strategy (Asserts vs Validate):
------------------------------------------
Development (assertions):
• Constructors use assert statements for structural checks
• assert values.ndim == 1, "values must be 1D"
• assert len(values) == length, "length mismatch"
• Assertions are removed with Python -O flag for production

Testing/Debugging (validate method):
• Use validate() for comprehensive checking
• Returns all errors at once (better debugging)
• Includes optional range validation (check_ranges=True)
• Use in tests, development, and untrusted input

Production:
• Run with python -O to remove assertions (faster)
• Skip validate() calls unless validating external input

Philosophy:
• Fast by default (no validation overhead in production)
• Asserts catch programmer errors during development
• validate() catches data errors from external sources
• Use asserts for "should never happen" conditions
• Use validate() for "might happen with bad input" conditions

Performance Contracts:
----------------------
Fast (O(1) or O(n)):
• Property access: values, scores, valid_mask, valid_count
• Indexing: feature[element]
• Cached properties (after first access)
• Array operations: feature.values[mask]

Moderate (Python conversion):
• get(element, fill), get_score(element)

Slower (iteration or allocation):
• get_values(elements), get_scores(elements)
• validate() with check_ranges=True

Guidance:
• Use raw access in tight loops or vectorized operations
• Use python-friendly access for clarity and safety
• Cache results of get_values() if accessing multiple times
• Avoid repeated validate() calls in production code

Validation Philosophy:
----------------------
• Construction is fast by default (minimal assertions)
• Assertions removed with -O flag for production performance
• Use validate() for debugging and testing
• Validation returns all errors at once (better debugging)
• Optional range checking (disabled for infinite bounds)

Abstract Methods (must implement in subclasses):
-------------------------------------------------
Structure:
  • enum() -> type[IntEnum]        Feature element enum
  • range() -> tuple[float, float] Valid value range
  • __len__() -> int                       Number of elements

Data Access:
  • values: np.ndarray                     Raw values array
  • scores: np.ndarray                     Confidence scores
  • valid_mask: np.ndarray                 Boolean validity mask
  • valid_count: int                       Count of valid elements

Element Access:
  • get_score(element) -> float            Single confidence score
  • get_scores(elements) -> list[float]    Multiple scores

Utilities:
  • validate(check_ranges) -> tuple[bool, str|None]  Validation
  • __repr__() -> str                      String representation

Range Constants:
----------------
Convenience constants for common ranges (optional, can be used in range()):

• NORMALIZED_RANGE = (0.0, 1.0)           For normalized values (probabilities, ratios)
• POSITIVE_RANGE = (0.0, np.inf)          For non-negative values (distances, heights)
• UNBOUNDED_RANGE = (-np.inf, np.inf)     For any real value (coordinates, differences)
• PI_RANGE = (0.0, np.pi)                 For angles [0, π]
• TWO_PI_RANGE = (0.0, 2*np.pi)           For angles [0, 2π]
• SYMMETRIC_PI_RANGE = (-np.pi, np.pi)    For angles [-π, π]

Infinite bounds (±np.inf) disable range validation for that bound.
=============================================================================
"""
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

    All pose features share structure definition, raw data arrays,
    validity tracking, and common constructor patterns.

    See module docstring for full API reference and design philosophy.
    """

    # ========== STRUCTURE ==========

    @classmethod
    @abstractmethod
    def enum(cls) -> type[FeatureEnum]:
        """Feature enum defining the elements of this feature.

        Returns:
            IntEnum subclass defining elements for this feature type.

        Note: For runtime-configured features, generate the enum at
              configuration time (see Similarity for example).
        """
        ...

    @classmethod
    @abstractmethod
    def length(cls) -> int:
        """Number of elements in this feature type.

        Returns:
            Fixed length for this feature type.

        Standard implementation: return len(cls.enum())
        """
        ...

    @classmethod
    @abstractmethod
    def range(cls) -> tuple[float, float]:
        """Define valid value range for this feature type.

        Returns:
            (min_value, max_value) tuple. Use module constants.
        """
        ...

    @classmethod
    def display_range(cls) -> tuple[float, float]:
        """Display range for visualization. Defaults to range().

        Override in subclasses where the validation range is too wide
        or unbounded for meaningful display (e.g. AngleVelocity, BBox, Points2D).
        """
        ...

    @classmethod
    @abstractmethod
    def create_dummy(cls) -> "BaseFeature":
        """Create a dummy instance with NaN values and zero scores.

        Returns:
            Instance with NaN values and zero scores for initialization.
        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Number of elements in this feature."""
        ...

    # ========== RAW DATA ACCESS ==========

    @property
    @abstractmethod
    def values(self) -> np.ndarray:
        """Feature values array (read-only, shape varies by type)."""
        ...

    @property
    @abstractmethod
    def scores(self) -> np.ndarray:
        """Confidence scores array (read-only, n_elements)."""
        ...

    # ========== VALIDITY ==========

    @property
    @abstractmethod
    def valid_mask(self) -> np.ndarray:
        """Boolean mask of valid elements (n_elements)."""
        ...

    @property
    @abstractmethod
    def valid_count(self) -> int:
        """Number of elements with valid data."""
        ...

    # ========== SCORE ACCESS ==========

    @abstractmethod
    def get_score(self, element: FeatureEnum | int) -> float:
        """Get confidence score for a single element (Python float)."""
        ...

    @abstractmethod
    def get_scores(self, elements: list[FeatureEnum | int]) -> list[float]:
        """Get confidence scores for multiple elements (Python list)."""
        ...

    # ========== VALIDATION ==========

    @abstractmethod
    def validate(self, check_ranges: bool = True) -> tuple[bool, str | None]:
        """Validate feature data integrity.

        Args:
            check_ranges: Whether to perform range validation on values.

        Returns:
            Tuple of (is_valid, error_message if invalid).
        """
        ...

    # ========== REPRESENTATION ==========

    @abstractmethod
    def __repr__(self) -> str:
        """String representation for debugging."""
        ...