"""Renders centered and rotated camera view based on pose anchor points."""

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.render.layers.LayerBase import LayerBase
from modules.render.layers.renderers import CamImageRenderer
from modules.render.layers.centre.CentreGeometry import CentreGeometry
from modules.render.shaders import Blend, DrawRoi
from modules.utils.PointsAndRects import Rect

# GL
from modules.gl import Fbo, SwapFbo, Texture


class CentreCamLayer(LayerBase):
    """Renders camera image cropped and rotated around pose anchor points.

    Reads anchor geometry from AnchorPointCalculator and applies DrawRoi
    shader followed by temporal blending.
    """

    def __init__(self, anchor_calc: CentreGeometry, cam_image: CamImageRenderer) -> None:
        self._anchor_calc: CentreGeometry = anchor_calc
        self._cam_image: CamImageRenderer = cam_image

        # FBOs
        self._cam_fbo: Fbo = Fbo()
        self._cam_blend_fbo: SwapFbo = SwapFbo()

        # Shaders
        self._roi_shader = DrawRoi()
        self._blend_shader = Blend()

        # Configuration
        self.blend_factor: float = 0.5

    @property
    def texture(self) -> Texture:
        """Output texture for external use."""
        return self._cam_blend_fbo.texture

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._cam_fbo.allocate(width, height, internal_format)
        self._cam_blend_fbo.allocate(width, height, internal_format)
        self._roi_shader.allocate()
        self._blend_shader.allocate()

    def deallocate(self) -> None:
        self._cam_fbo.deallocate()
        self._cam_blend_fbo.deallocate()
        self._roi_shader.deallocate()
        self._blend_shader.deallocate()

    def update(self) -> None:
        """Render camera crop using anchor geometry."""
        # Disable blending during FBO rendering
        glDisable(GL_BLEND)
        glColor4f(1.0, 1.0, 1.0, 1.0)

        self._cam_fbo.clear(0.0, 0.0, 0.0, 0.0)

        # Check if valid anchor data exists
        if not self._anchor_calc.has_pose:
            self._cam_blend_fbo.clear(0.0, 0.0, 0.0, 0.0)
            self._cam_blend_fbo.swap()
            self._cam_blend_fbo.clear(0.0, 0.0, 0.0, 0.0)
            glEnable(GL_BLEND)
            return

        # Render camera image with ROI from anchor calculator
        cam_aspect: float = self._cam_image.texture.width / self._cam_image.texture.height
        self._roi_shader.use(
            self._cam_fbo.fbo_id,
            self._cam_image.texture.tex_id,
            self._anchor_calc.cam_crop_roi,
            self._anchor_calc.cam_rotation,
            self._anchor_calc.anchor_top_tex,
            cam_aspect,
            False,
            True
        )

        # Temporal blending
        self._cam_blend_fbo.swap()
        self._blend_shader.use(
            self._cam_blend_fbo.fbo_id,
            self._cam_blend_fbo.back_tex_id,
            self._cam_fbo.tex_id,
            self.blend_factor
        )

        glEnable(GL_BLEND)
