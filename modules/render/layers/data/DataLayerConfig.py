# Standard library imports
from dataclasses import dataclass

# Local application imports
from modules.ConfigBase import ConfigBase, config_field
from modules.DataHub import Stage
from modules.pose.Frame import FrameField


# Individual color constants
POSE_COLOR_LEFT:    tuple[float, float, float, float] = (1.0, 0.5, 0.0, 1.0) # Orange
POSE_COLOR_RIGHT:   tuple[float, float, float, float] = (0.0, 1.0, 1.0, 1.0) # Cyan
POSE_COLOR_CENTER:  tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0) # White

SIMILARITY_COLOR_LOW:    tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0) # Red
SIMILARITY_COLOR_MID:    tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0) # Green
SIMILARITY_COLOR_HIGH:   tuple[float, float, float, float] = (1.0, 1.0, 0.0, 1.0) # Yellow

# Color lists for different feature types
DEFAULT_COLORS: list[tuple[float, float, float, float]] = [
    POSE_COLOR_LEFT,   # Orange
    POSE_COLOR_RIGHT,  # Cyan
]

SIMILARITY_COLORS: list[tuple[float, float, float, float]] = [
    SIMILARITY_COLOR_LOW,   # Red
    SIMILARITY_COLOR_MID,   # Green
    SIMILARITY_COLOR_HIGH,  # Yellow
]

BBOX_COLORS: list[tuple[float, float, float, float]] = [
    (1.0, 0.0, 0.0, 1.0),  # Red
    (0.0, 1.0, 0.0, 1.0),  # Green
    (1.0, 0.5, 0.0, 1.0),  # Orange
    (1.0, 1.0, 0.0, 1.0),  # Yellow
]


# FrameField → color list lookup
FEATURE_COLORS: dict[FrameField, list[tuple[float, float, float, float]]] = {
    FrameField.bbox:            BBOX_COLORS,
    FrameField.angles:          DEFAULT_COLORS,
    FrameField.angle_vel:       DEFAULT_COLORS,
    FrameField.angle_motion:    DEFAULT_COLORS,
    FrameField.angle_sym:       DEFAULT_COLORS,
    FrameField.similarity:      SIMILARITY_COLORS,
    FrameField.leader:          SIMILARITY_COLORS,
    FrameField.motion_gate:     SIMILARITY_COLORS,
}


# Scalar fields only - for data visualization dropdown
SCALAR_FIELDS: list[FrameField] = FrameField.get_scalar_fields()


@dataclass
class DataLayerConfig(ConfigBase):
    """Unified configuration for data visualization layers."""
    active: bool =              config_field(False)
    feature_field: FrameField = config_field(FrameField.angle_motion, choices=SCALAR_FIELDS)
    stage: Stage =              config_field(Stage.SMOOTH)

    line_width: float =         config_field(3.0)
    line_smooth: float =        config_field(1.0)

    use_scores: bool =          config_field(False)
    render_labels: bool =       config_field(True)

    # Not a config_field - stays outside ConfigBase watch system
    colors: list[tuple[float, float, float, float]] | None = None
