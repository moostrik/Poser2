"""Renders pose lines using pre-transformed points from AnchorPointCalculator."""

# Standard library imports
import numpy as np

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl import Fbo, Texture, clear_color
from modules.render.layers.LayerBase import LayerBase
from modules.render.layers.centre.CentreGeometry import CentreGeometry
from modules.render.shaders import PosePointLines, DrawCircles
from modules.utils.HotReloadMethods import HotReloadMethods


class CentrePoseLayer(LayerBase):
    """Renders pose keypoint lines in crop space."""

    def __init__(self, geometry: CentreGeometry,
                 line_width: float = 4.0, line_smooth: float = 2.0, use_scores: bool = False,
                 color: tuple[float, float, float, float] | None = None, draw_anchors: bool = True) -> None:
        self._geometry: CentreGeometry = geometry
        self._fbo: Fbo = Fbo()
        self._shader: PosePointLines = PosePointLines()
        self._circle_shader: DrawCircles = DrawCircles()

        self.line_width: float = line_width
        self.line_smooth: float = line_smooth
        self.use_scores: bool = use_scores
        self.color: tuple[float, float, float, float] | None = color
        self.draw_anchors: bool = draw_anchors

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

        transformed_points = self._geometry.transformed_points
        if transformed_points is None:
            return

        line_width: float = 1.0 / self._fbo.height * self.line_width
        line_smooth: float = 1.0 / self._fbo.height * self.line_smooth
        anchor_size: float = line_width * 4.0
        anchor_smooth: float = line_smooth
        anchor_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)

        # Create anchor circle positions
        positions = [
            (self._geometry.target_top.x, 1.0 - self._geometry.target_top.y),
            (self._geometry.target_bottom.x, 1.0 - self._geometry.target_bottom.y)
        ]

        aspect_ratio = self._fbo.width / self._fbo.height if self._fbo.height > 0 else 1.0

        self._fbo.begin()
        clear_color()
        self._shader.use(transformed_points, line_width, line_smooth, self.color, self.use_scores)
        if self.draw_anchors:
            self._circle_shader.use(positions, anchor_size, anchor_smooth, anchor_color, aspect_ratio)
        self._fbo.end()