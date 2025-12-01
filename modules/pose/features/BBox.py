"""
=============================================================================
BBOXFEATURE API REFERENCE
=============================================================================

Concrete implementation of BaseScalarFeature for bounding box storage.

Use for: tracked bounding boxes, region of interest, smoothed viewport data.

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
  • BBoxFeature(values, scores)           → Direct (fast, no validation)
  • BBoxFeature.create_empty()            → All NaN values, zero scores
  • BBoxFeature.from_rect(rect, ...)      → Create from Rect object

Validation:
  • Asserts in constructors (removed with -O flag for production)
  • validate() method for debugging/testing/untrusted input
  • Fast by default, validate only when needed

Performance:
  Fast:     Property access, indexing, cached properties, array ops
  Moderate: get(), get_score() (Python conversion)
  Slow:     get_values(), get_scores() (iteration), validate()

Inherited from BaseScalarFeature:
==================================

Structure:
----------
Each element has:
  • A scalar value (float) - may be NaN for invalid/missing data
  • A confidence score [0.0, 1.0]

Storage:
  • values: np.ndarray, shape (4,), dtype float32, unbounded range
  • scores: np.ndarray, shape (4,), dtype float32

Properties:
-----------
  • values: np.ndarray                             All bbox values (4,)
  • scores: np.ndarray                             All confidence scores (4,)
  • valid_mask: np.ndarray                         Boolean validity mask (4,)
  • valid_count: int                               Number of valid values
  • len(feature): int                              Total number of elements (4)

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
  • BBoxFeature.create_empty() -> BBoxFeature          All NaN values, zero scores
  • BBoxFeature.from_rect(rect, scores?) -> BBoxFeature      Create from Rect object

Validation:
-----------
  • feature.validate(check_ranges=True) -> tuple[bool, str|None]
      Returns (is_valid, error_message)

Implemented Methods (from BaseScalarFeature):
----------------------------------------------
Structure:
  • enum() -> type[BBoxProperty]           Returns BBoxProperty enum (IMPLEMENTED)
  • range() -> tuple[float, float]         Returns (-inf, inf) (IMPLEMENTED)

BBoxFeature-Specific:
=====================

BBoxProperty Enum (bounding box components):
--------------------------------------------
  • centre_x (0)  - Center X coordinate
  • centre_y (1)  - Center Y coordinate
  • width (2)     - Bounding box width
  • height (3)    - Bounding box height

Rect Conversion:
----------------
  • feature.to_rect() -> Rect
      Convert stored bbox values to Rect object
      Fast conversion - just arithmetic, no caching needed

Factory from Rect:
------------------
  • BBoxFeature.from_rect(rect, scores=None) -> BBoxFeature
      Create from Rect object
      - Auto-generates scores if None (1.0 for valid, 0.0 for NaN)
      - Converts 0 dimensions to NaN (0 means "no dimension")

Value Constraints:
------------------
  • Width and height cannot be 0.0 (use NaN for missing dimensions)
  • Centre coordinates can be any value (unbounded)
  • All values are smoothed independently (aspect ratio may change)

Use Cases:
----------
1. Tracker Output:
   >>> rect = tracker.get_bbox()
   >>> bbox = BBoxFeature.from_rect(rect)

2. Smoothing Over Time:
   >>> # Smooth each component independently
   >>> smoothed_cx = smooth(bbox[BBoxProperty.centre_x])
   >>> smoothed_cy = smooth(bbox[BBoxProperty.centre_y])
   >>> smoothed_w = smooth(bbox[BBoxProperty.width])
   >>> smoothed_h = smooth(bbox[BBoxProperty.height])

3. Convert Back to Rect:
   >>> rect = bbox.to_rect()
   >>> # Apply aspect ratio downstream if needed
   >>> display_rect = apply_aspect_ratio(rect, 16/9, mode='fit')

Design Notes:
-------------
- BBoxFeature stores smoothed region data, NOT a fixed-aspect viewport
- Aspect ratio may change through filtering (this is intentional)
- Downstream consumers apply target aspect ratios as needed
- Width/height are smoothed independently for best noise reduction
- Zero values not allowed (semantic meaning conflicts with NaN)

Notes:
------
- All values are single floats (scalar per element)
- Invalid values are NaN with score 0.0
- Arrays are read-only after construction (immutable)
- Use validate() for debugging, not in production loops
- Constructor takes ownership - caller must not modify arrays after passing
=============================================================================
"""
from enum import IntEnum

import numpy as np

from modules.pose.features.base.BaseScalarFeature import BaseScalarFeature
from modules.utils.PointsAndRects import Rect


class BBoxElement(IntEnum):
    centre_x = 0
    centre_y = 1
    width = 2
    height = 3


# Constants
BBOX_ELEMENT_NAMES: list[str] = [e.name for e in BBoxElement]
BBOX_NUM_ELEMENTS: int = len(BBoxElement)


class BBox(BaseScalarFeature[BBoxElement]):
    """Bounding box feature storing center position, width, and height.

    Use downstream code to apply target aspect ratios as needed.
    """

    def __init__(self, values: np.ndarray, scores: np.ndarray) -> None:
        """Initialize BBoxFeature.

        Args:
            values: [centre_x, centre_y, width, height]
            scores: Confidence scores for each value
        """

        assert values[BBoxElement.width] != 0.0, "Width cannot be zero, use NaN for 'no width'"
        assert values[BBoxElement.height] != 0.0, "Height cannot be zero, use NaN for 'no height'"

        super().__init__(values, scores)

    # ========== ABSTRACT METHOD IMPLEMENTATIONS ==========

    @classmethod
    def enum(cls) -> type[IntEnum]:
        return BBoxElement

    @classmethod
    def range(cls) -> tuple[float, float]:
        # return (-np.inf, np.inf)
        return (-2.0, 2.0)

    # ========== RECT CONVERSION ==========

    def to_rect(self) -> Rect:
        """Convert bounding box to Rect using stored dimensions.

        Returns:
            Rect representing the bounding box
        """
        cx = self[BBoxElement.centre_x]
        cy = self[BBoxElement.centre_y]
        w = self[BBoxElement.width]
        h = self[BBoxElement.height]

        return Rect(
            x=cx - w / 2,
            y=cy - h / 2,
            width=w,
            height=h
        )

    # ========== FACTORY METHODS ==========

    @classmethod
    def from_rect(cls, rect: Rect, scores: np.ndarray | None = None) -> 'BBox':
        """Create BBoxFeature from Rect (stores all dimensions).

        Args:
            rect: Rectangle defining bounding box
            scores: Optional confidence scores (auto-generated if None)

        Returns:
            BBoxFeature storing center, width, and height
        """
        # Convert 0 dimensions to NaN (0 means "no dimension")
        w = rect.width if rect.width != 0.0 else np.nan
        h = rect.height if rect.height != 0.0 else np.nan

        values = np.array([rect.center.x,
                          rect.center.y,
                          w,
                          h], dtype=np.float32)

        if scores is None:
            scores = np.where(np.isnan(values), 0.0, 1.0).astype(np.float32)

        return cls(values, scores)

    @classmethod
    def create_dummy(cls) -> 'BBox':
        """Create empty feature with all NaN values and zero scores."""
        values = np.full(len(cls.enum()), np.nan, dtype=np.float32)
        scores = np.zeros(len(cls.enum()), dtype=np.float32)
        return cls(values, scores)


    # ========== REPRESENTATION ==========

    def __repr__(self) -> str:
        if self.valid_count == len(self):
            cx = self[BBoxElement.centre_x]
            cy = self[BBoxElement.centre_y]
            w = self[BBoxElement.width]
            h = self[BBoxElement.height]
            return f"BBoxFeature(centre=({cx:.1f}, {cy:.1f}), size={w:.1f}x{h:.1f})"
        else:
            return f"BBoxFeature(valid={self.valid_count}/{len(self)})"
