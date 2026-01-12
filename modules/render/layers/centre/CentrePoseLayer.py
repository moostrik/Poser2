"""Renders pose lines using pre-transformed points from AnchorPointCalculator."""

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl import Fbo, Texture
from modules.render.layers.LayerBase import LayerBase, Rect
from modules.render.layers.centre.CentreGeometry import CentreGeometry
from modules.render.shaders import PosePointLines
from modules.utils.HotReloadMethods import HotReloadMethods


class CentrePoseLayer(LayerBase):
    """Renders pose keypoint lines in crop space."""

    def __init__(self, anchor_calc: CentreGeometry,
                 line_width: float = 4.0, line_smooth: float = 2.0, use_scores: bool = False,
                 color: tuple[float, float, float, float] | None = None) -> None:
        self._anchor_calc: CentreGeometry = anchor_calc
        self._fbo: Fbo = Fbo()
        self._shader: PosePointLines = PosePointLines()

        self.line_width: float = line_width
        self.line_smooth: float = line_smooth
        self.use_scores: bool = use_scores
        self.color: tuple[float, float, float, float] | None = color

        HotReloadMethods(self.__class__, True, True)

    @property
    def texture(self) -> Texture:
        """Output texture for external use."""
        return self._fbo.texture

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._fbo.allocate(width, height, internal_format)
        self._shader.allocate()

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._shader.deallocate()

    def update(self) -> None:
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        self._fbo.clear(0.0, 0.0, 0.0, 0.0)

        transformed_points = self._anchor_calc.transformed_points
        if transformed_points is None:
            return

        line_width: float = 1.0 / self._fbo.height * self.line_width
        line_smooth: float = 1.0 / self._fbo.height * self.line_smooth

        self._shader.use(self._fbo, transformed_points,
                        line_width=line_width, line_smooth=line_smooth,
                        color=self.color, use_scores=self.use_scores)