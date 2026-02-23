"""Renders pose lines using pre-transformed points from AnchorPointCalculator."""

# Standard library imports
import numpy as np

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.settings import Setting, BaseSettings
from modules.gl import Fbo, Texture, clear_color
from modules.render.layers.LayerBase import LayerBase
from modules.render.layers.centre.CentreGeometry import CentreGeometry
from modules.render.shaders import PosePointLines, DrawCircles
from modules.render.color_settings import ColorSettings
from modules.utils.HotReloadMethods import HotReloadMethods


class CentrePoseSettings(BaseSettings):
    """Configuration for CentrePoseLayer pose line rendering."""
    line_width:     Setting[float] = Setting(3.0, min=0.5, max=10.0, description="Pose line thickness")
    line_smooth:    Setting[float] = Setting(0.0, min=0.0, max=5.0, description="Line anti-aliasing radius")
    use_scores:     Setting[bool]  = Setting(False, description="Color lines by confidence scores")
    draw_anchors:   Setting[bool]  = Setting(False, description="Show anchor points as circles")


class CentrePoseLayer(LayerBase):
    """Renders pose keypoint lines in crop space."""

    def __init__(self, cam_id: int, geometry: CentreGeometry, settings: CentrePoseSettings, color_settings: ColorSettings) -> None:
        self._cam_id: int = cam_id
        self._geometry: CentreGeometry = geometry
        self._fbo: Fbo = Fbo()
        self._shader: PosePointLines = PosePointLines()
        self._circle_shader: DrawCircles = DrawCircles()

        # Configuration
        self.settings: CentrePoseSettings = settings or CentrePoseSettings()
        self._color_settings: ColorSettings = color_settings

        # HotReloadMethods(self.__class__, True, True)

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

        line_width: float = 1.0 / self._fbo.height * self.settings.line_width
        line_smooth: float = 1.0 / self._fbo.height * self.settings.line_smooth
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
        color = self._color_settings.track_colors[self._cam_id % len(self._color_settings.track_colors)].to_tuple()
        self._shader.use(transformed_points, line_width, line_smooth, color, self.settings.use_scores)
        if self.settings.draw_anchors:
            self._circle_shader.use(positions, anchor_size, anchor_smooth, anchor_color, aspect_ratio)
        self._fbo.end()