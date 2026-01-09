"""Renders centered and rotated optical flow visualization."""

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.render.layers.LayerBase import TextureLayer
from modules.render.layers.renderers import DenseFlowRenderer
from modules.render.layers.centre.CentreGeometry import CentreGeometry
from modules.render.shaders import DrawRoi
from modules.utils.PointsAndRects import Rect

# GL
from modules.gl import Fbo, Texture


class CentreDenseFlowLayer(TextureLayer):
    """Renders optical flow visualization cropped and rotated around pose anchor points.

    Reads anchor geometry from CentreGeometry and applies DrawRoi shader to flow visualization.
    """

    def __init__(self, anchor_calc: CentreGeometry, flow_renderer: DenseFlowRenderer) -> None:
        self._anchor_calc: CentreGeometry = anchor_calc
        self._flow_renderer: DenseFlowRenderer = flow_renderer

        # FBO
        self._flow_fbo: Fbo = Fbo()

        # Shader
        self._roi_shader = DrawRoi()

    @property
    def texture(self) -> Texture:
        """Output texture for external use."""
        return self._flow_fbo

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._flow_fbo.allocate(width, height, internal_format)
        self._roi_shader.allocate()

    def deallocate(self) -> None:
        self._flow_fbo.deallocate()
        self._roi_shader.deallocate()

    def update(self) -> None:
        """Render flow crop using anchor geometry."""
        # Disable blending during FBO rendering
        glDisable(GL_BLEND)
        glColor4f(1.0, 1.0, 1.0, 1.0)

        self._flow_fbo.clear(0.0, 0.0, 0.0, 0.0)

        # Check if valid anchor data exists
        if not self._anchor_calc.has_pose:
            glEnable(GL_BLEND)
            return

        # Check if flow renderer has valid data
        if not self._flow_renderer._fbo.allocated:
            glEnable(GL_BLEND)
            return

        # Render flow with ROI from anchor calculator (bbox-space geometry, like mask)
        self._roi_shader.use(
            self._flow_fbo.fbo_id,
            self._flow_renderer.texture.tex_id,
            self._anchor_calc.bbox_crop_roi,
            self._anchor_calc.bbox_rotation,
            self._anchor_calc.bbox_rotation_center,
            self._anchor_calc.bbox_aspect,
            False,
            False
        )

        glEnable(GL_BLEND)
