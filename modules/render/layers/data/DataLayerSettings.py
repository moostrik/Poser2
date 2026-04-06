# Standard library imports
import math
from enum import IntEnum, auto

# Local application imports
from modules.settings import Field, Settings
from modules.data_hub import Stage
from modules.pose.features import FEATURES
from modules.pose.features.base.BaseScalarFeature import BaseScalarFeature


# ---------------------------------------------------------------------------
#  Auto-discover visualizable scalar features
# ---------------------------------------------------------------------------

_VISUALIZABLE = sorted(
    [f for f in FEATURES
     if issubclass(f, BaseScalarFeature)
     and math.isfinite(f.display_range()[0])
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
from modules.pose.features import Similarity, LeaderScore, MotionGate
TRACK_COLOR_FEATURES: set[type[BaseScalarFeature]] = {Similarity, LeaderScore, MotionGate}


class LayerMode(IntEnum):
    """Display mode for a data layer slot. NONE disables rendering."""
    NONE =      0
    FRAME =     auto()
    WINDOW =    auto()


class DataLayerSettings(Settings):
    """Unified configuration for data visualization layers."""
    mode:           Field[LayerMode]            = Field(LayerMode.WINDOW)
    feature_field:  Field[ScalarFeatureSelect]  = Field(ScalarFeatureSelect.AngleMotion)  # type: ignore[attr-defined]
    stage:          Field[Stage]                = Field(Stage.SMOOTH)

    line_width:     Field[float] = Field(3.0)
    line_smooth:    Field[float] = Field(1.0)

    use_scores:     Field[bool]  = Field(False)
    render_labels:  Field[bool]  = Field(True)
    use_history_color: Field[bool] = Field(False)
