# Standard library imports
import math

# Third-party imports
from OpenGL.GL import *  # type: ignore

# Local application imports
from modules.DataHub import DataHub, DataHubType
from modules.gl import Fbo, clear_color, draw_box_string, text_init
from modules.gl.Shader import Shader
from modules.pose.features.Angles import ANGLE_NUM_LANDMARKS, ANGLE_LANDMARK_NAMES
from modules.render.layers.LayerBase import Rect
from modules.render.shaders.window import PoseAngleVelocityWindow, WindowShaderBase
from ..data.Colors import POSE_COLOR_LEFT, POSE_COLOR_RIGHT
from .WindowLayerBase import WindowLayerBase


class AngleVelWindowLayer(WindowLayerBase):
    """Visualizes angle velocity window for a single track.

    Displays angular velocity values as horizontal lines (one per angle element),
    with time flowing left-to-right. Sources data from AngleVelocityWindowTracker
    via DataHub per-track windows.
    """

    def __init__(self, track_id: int, data_hub: DataHub) -> None:
        super().__init__(track_id, data_hub, DataHubType.angle_vel_window)

    def get_shader(self) -> WindowShaderBase:
        """Return PoseAngleVelocityWindow shader instance."""
        return PoseAngleVelocityWindow()

    def get_feature_names(self) -> list[str]:
        """Return angle landmark names."""
        return ANGLE_LANDMARK_NAMES

    def get_display_range(self) -> tuple[float, float]:
        """Return ±π range for angular velocity."""
        return (-math.pi, math.pi)

    def render_labels_static(self, fbo: Fbo) -> None:
        """Render angle labels overlay."""
        text_init()

        rect = Rect(0, 0, fbo.width, fbo.height)

        fbo.begin()
        clear_color()

        angle_num: int = ANGLE_NUM_LANDMARKS
        step: float = rect.height / angle_num
        colors: list[tuple[float, float, float, float]] = [
            (*POSE_COLOR_LEFT, 1.0),
            (*POSE_COLOR_RIGHT, 1.0)
        ]

        for i in range(angle_num):
            string: str = ANGLE_LANDMARK_NAMES[i]
            x: int = int(rect.x + 10)
            y: int = int(rect.y + rect.height - (rect.height - (i + 0.5) * step) - 9)
            clr: int = i % 2

            draw_box_string(x, y, string, colors[clr], (0.0, 0.0, 0.0, 0.3))  # type: ignore

        fbo.end()
