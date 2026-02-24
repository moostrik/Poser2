# Standard library imports
from enum import IntEnum, auto

# Local application imports
from modules.settings import Field, Settings
from modules.DataHub import Stage
from modules.pose.Frame import FrameField, ScalarFrameField
from modules.render.color_settings import ColorSettings


class LayerMode(IntEnum):
    """Display mode for a data layer slot. NONE disables rendering."""
    NONE =      0
    FRAME =     auto()
    WINDOW =    auto()


# Fields that use track colors (the rest use DEFAULT_COLORS)
_TRACK_COLOR_FIELDS: set[ScalarFrameField] = {
    ScalarFrameField.similarity,
    ScalarFrameField.leader,
    ScalarFrameField.motion_gate,
}


class DataLayerSettings(Settings):
    """Unified configuration for data visualization layers."""
    mode:           Field[LayerMode]        = Field(LayerMode.WINDOW)
    feature_field:  Field[ScalarFrameField] = Field(ScalarFrameField.angle_motion)
    stage:          Field[Stage]            = Field(Stage.SMOOTH)

    line_width:     Field[float] = Field(3.0)
    line_smooth:    Field[float] = Field(1.0)

    use_scores:     Field[bool]  = Field(False)
    render_labels:  Field[bool]  = Field(True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Not a Setting — stays outside the descriptor system
        self._colors: list[tuple[float, float, float, float]] | None = None
        self._color_settings: ColorSettings | None = None

    @property
    def colors(self) -> list[tuple[float, float, float, float]] | None:
        return self._colors

    @colors.setter
    def colors(self, value: list[tuple[float, float, float, float]] | None) -> None:
        self._colors = value

    @property
    def color_settings(self) -> ColorSettings | None:
        return self._color_settings

    @color_settings.setter
    def color_settings(self, value: ColorSettings) -> None:
        self._color_settings = value

    def get_colors(self) -> list[tuple[float, float, float, float]]:
        """Resolve colors for the current feature field.

        Priority: explicit _colors override → track colors from ColorSettings → default palette.
        """
        if self._colors is not None:
            return self._colors
        if self._color_settings is not None:
            if self.feature_field in _TRACK_COLOR_FIELDS:
                return self._color_settings.track_color_tuples
            return self._color_settings.default_color_tuples
        return [(1.0, 0.5, 0.0, 1.0), (0.0, 1.0, 1.0, 1.0)]  # hardcoded fallback (should not be reached)
