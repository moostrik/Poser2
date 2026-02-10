# Standard library imports
from dataclasses import dataclass

# Local application imports
from modules.ConfigBase import ConfigBase, config_field
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


@dataclass
class DataLayerConfig(ConfigBase):
    """Unified configuration for data visualization layers. Active state is per-layer instance."""
    feature_field: ScalarFrameField =   config_field(ScalarFrameField.angle_motion)
    stage: Stage =                      config_field(Stage.SMOOTH)

    line_width: float =         config_field(3.0)
    line_smooth: float =        config_field(1.0)

    use_scores: bool =          config_field(False)
    render_labels: bool =       config_field(True)

    # Not a config_field - stays outside ConfigBase watch system
    colors: list[tuple[float, float, float, float]] | None = None
