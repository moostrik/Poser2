"""Renders pose lines using pre-transformed points from AnchorPointCalculator."""

# Standard library imports
import numpy as np
from dataclasses import dataclass

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.ConfigBase import ConfigBase, config_field
from modules.gl import Fbo, Texture, clear_color
from modules.render.layers.LayerBase import LayerBase
from modules.render.layers.centre.CentreGeometry import CentreGeometry
from modules.render.shaders import PosePointLines, DrawCircles
from modules.utils.HotReloadMethods import HotReloadMethods


@dataclass
class CentrePoseConfig(ConfigBase):
    """Configuration for CentrePoseLayer pose line rendering."""
    line_width: float = config_field(4.0, min=0.5, max=10.0, description="Pose line thickness")
    line_smooth: float = config_field(2.0, min=0.0, max=5.0, description="Line anti-aliasing radius")
    use_scores: bool = config_field(False, description="Color lines by confidence scores")
    draw_anchors: bool = config_field(True, description="Show anchor points as circles")


class CentrePoseLayer(LayerBase):
    """Renders pose keypoint lines in crop space."""

    def __init__(self, geometry: CentreGeometry, color: tuple[float, float, float, float] | None = None, config: CentrePoseConfig | None = None) -> None:
        self._geometry: CentreGeometry = geometry
        self._fbo: Fbo = Fbo()
        self._shader: PosePointLines = PosePointLines()
        self._circle_shader: DrawCircles = DrawCircles()

        # Configuration
        self.config: CentrePoseConfig = config or CentrePoseConfig()
        self.color: tuple[float, float, float, float] | None = color

        HotReloadMethods(self.__class__, True, True)

    @property
    def texture(self) -> Texture:
        """Output texture for external use."""
        return self._fbo.texture

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._fbo.allocate(width, height, internal_format)
        self._shader.allocate()
        self._circle_shader.allocate()

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._shader.deallocate()
        self._circle_shader.deallocate()

    def update(self) -> None:
        if self._geometry.lost:
            self._fbo.clear()
            return

        transformed_points = self._geometry.crop_pose_points
        if transformed_points is None:
            return

        line_width: float = 1.0 / self._fbo.height * self.config.line_width
        line_smooth: float = 1.0 / self._fbo.height * self.config.line_smooth
        anchor_size: float = line_width * 4.0
        anchor_smooth: float = line_smooth
        anchor_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)

        # Create anchor circle positions from config
        positions = [
            (self._geometry.config.target_top_x, 1.0 - self._geometry.config.target_top_y),
            (self._geometry.config.target_bottom_x, 1.0 - self._geometry.config.target_bottom_y)
        ]

        aspect_ratio = self._fbo.width / self._fbo.height if self._fbo.height > 0 else 1.0

        self._fbo.begin()
        clear_color()
        self._shader.use(transformed_points, line_width, line_smooth, self.color, self.config.use_scores)
        if self.config.draw_anchors:
            self._circle_shader.use(positions, anchor_size, anchor_smooth, anchor_color, aspect_ratio)
        self._fbo.end()