# Standard library imports
import math
from enum import IntEnum, auto
from typing import Protocol

# Third-party imports
import numpy as np

# Local application imports
from modules.settings import Field, BaseSettings
from modules.pose.features import SCALAR_FEATURES, TRACK_FEATURES, BaseScalarFeature, SYMMETRIC_PI_RANGE


# ---------------------------------------------------------------------------
#  Auto-discover visualizable scalar features
# ---------------------------------------------------------------------------

_VISUALIZABLE = sorted(
    [f for f in SCALAR_FEATURES
     if math.isfinite(f.display_range()[0])
     and math.isfinite(f.display_range()[1])],
    key=lambda c: c.__name__,
)

ScalarFeatureSelect = IntEnum(                          # type: ignore[misc]
    'ScalarFeatureSelect',
    {cls.__name__: i for i, cls in enumerate(_VISUALIZABLE)},
)
"""Selectable scalar features for data visualization layers.

Auto-built from all concrete BaseScalarFeature subclasses whose
display_range() has finite bounds.  Member names match class names
(e.g. AngleMotion, Angles, BBox).
"""

FEATURE_MAP: dict[ScalarFeatureSelect, type[BaseScalarFeature]] = {  # type: ignore[type-arg]
    ScalarFeatureSelect(i): cls for i, cls in enumerate(_VISUALIZABLE)  # type: ignore[misc]
}

# Features where each element represents a different track (use per-track colors)
TRACK_COLOR_FEATURES = TRACK_FEATURES


class AngleOffset(IntEnum):
    """Where the band seam sits when drawing ±π angles. NONE keeps it at ±π (the raised arm);
    HALF_PI shifts by -π/2 so it sits at -π/2 (the arm across the body) and the 0 → π arc runs
    unbroken through the centre; PI shifts by π so it sits at 0 (the hanging arm)."""
    NONE =      0
    HALF_PI =   auto()
    PI =        auto()


_OFFSET_RADIANS: dict[AngleOffset, float] = {
    AngleOffset.NONE:    0.0,
    AngleOffset.HALF_PI: -0.5 * np.pi,
    AngleOffset.PI:      np.pi,
}


def has_pi_range(feature_type: type[BaseScalarFeature]) -> bool:
    """True for wrapping angle features (range [-π, π]); the only ones `angle_offset` applies to."""
    return feature_type.range() == SYMMETRIC_PI_RANGE


def offset_angles(values: np.ndarray, offset: AngleOffset) -> np.ndarray:
    """Angles shifted by `offset` and wrapped back to [-π, π). NaN passes through; the input is not
    modified. NONE returns the input as is."""
    if offset == AngleOffset.NONE:
        return values
    shifted = np.mod(values + (_OFFSET_RADIANS[offset] + np.pi), 2.0 * np.pi) - np.pi
    return shifted.astype(values.dtype, copy=False)


class LayerMode(IntEnum):
    """Display mode for a data layer slot. NONE disables rendering."""
    NONE =      0
    FRAME =     auto()
    WINDOW =    auto()


class DataLayerConfig(Protocol):
    """Structural config a data layer reads. `DataLayerSettings` satisfies it, and so can an
    app-local settings class whose `feature_field` is a different IntEnum (keys into the
    layer's `feature_map`). Lets the generic layers serve app-owned dropdowns without
    depending on the module-global feature enum."""
    mode:              LayerMode
    feature_field:     IntEnum
    stage:             int
    line_width:        float
    line_smooth:       float
    use_scores:        bool
    render_labels:     bool
    use_history_color: bool
    angle_offset:      AngleOffset


class DataLayerSettings(BaseSettings):
    """Unified configuration for data visualization layers."""
    mode:           Field[LayerMode]            = Field(LayerMode.WINDOW)
    feature_field:  Field[ScalarFeatureSelect]  = Field(ScalarFeatureSelect.AngleMotion)  # type: ignore[attr-defined]
    stage:          Field[int]                  = Field(0)

    line_width:     Field[float] = Field(3.0)
    line_smooth:    Field[float] = Field(1.0)

    use_scores:     Field[bool]  = Field(False)
    render_labels:  Field[bool]  = Field(True)
    use_history_color: Field[bool] = Field(False)
    angle_offset:   Field[AngleOffset] = Field(AngleOffset.NONE, description="Shift ±π angles before drawing: HALF_PI puts the seam at -π/2, PI at 0")
