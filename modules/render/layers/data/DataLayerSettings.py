# Standard library imports

# Local application imports
from modules.settings import Setting, BaseSettings
from modules.DataHub import Stage
from modules.pose.Frame import FrameField, ScalarFrameField
from modules.render.layers.data.colors import DEFAULT_COLORS, TRACK_COLORS


# ScalarFrameField → color list lookup
FEATURE_COLORS: dict[ScalarFrameField, list[tuple[float, float, float, float]]] = {
    ScalarFrameField.bbox:            DEFAULT_COLORS,
    ScalarFrameField.angles:          DEFAULT_COLORS,
    ScalarFrameField.angle_vel:       DEFAULT_COLORS,
    ScalarFrameField.angle_motion:    DEFAULT_COLORS,
    ScalarFrameField.angle_sym:       DEFAULT_COLORS,
    ScalarFrameField.similarity:      TRACK_COLORS,
    ScalarFrameField.leader:          TRACK_COLORS,
    ScalarFrameField.motion_gate:     TRACK_COLORS,
}


class DataLayerSettings(BaseSettings):
    """Unified configuration for data visualization layers. Active state is per-layer instance."""
    feature_field:  Setting[ScalarFrameField] = Setting(ScalarFrameField.angle_motion)
    stage:          Setting[Stage]            = Setting(Stage.SMOOTH)

    line_width:     Setting[float] = Setting(3.0)
    line_smooth:    Setting[float] = Setting(1.0)

    use_scores:     Setting[bool]  = Setting(False)
    render_labels:  Setting[bool]  = Setting(True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Not a Setting — stays outside the descriptor system
        self._colors: list[tuple[float, float, float, float]] | None = None

    @property
    def colors(self) -> list[tuple[float, float, float, float]] | None:
        return self._colors

    @colors.setter
    def colors(self, value: list[tuple[float, float, float, float]] | None) -> None:
        self._colors = value
