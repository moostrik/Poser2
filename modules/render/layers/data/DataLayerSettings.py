# Standard library imports
from enum import IntEnum, auto

# Local application imports
from modules.settings import Field, Settings
from modules.data_hub import Stage
from modules.pose.frame import ScalarFrameField


class LayerMode(IntEnum):
    """Display mode for a data layer slot. NONE disables rendering."""
    NONE =      0
    FRAME =     auto()
    WINDOW =    auto()


class DataLayerSettings(Settings):
    """Unified configuration for data visualization layers."""
    mode:           Field[LayerMode]        = Field(LayerMode.WINDOW)
    feature_field:  Field[ScalarFrameField] = Field(ScalarFrameField.angle_motion)
    stage:          Field[Stage]            = Field(Stage.SMOOTH)

    line_width:     Field[float] = Field(3.0)
    line_smooth:    Field[float] = Field(1.0)

    use_scores:     Field[bool]  = Field(False)
    render_labels:  Field[bool]  = Field(True)
    use_history_color: Field[bool] = Field(False)
