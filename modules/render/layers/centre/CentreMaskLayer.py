"""Renders centered and rotated mask with temporal blending and blur."""

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.render.layers.LayerBase import LayerBase
from modules.render.layers.renderers import CamMaskRenderer
from modules.render.layers.centre.CentreGeometry import CentreGeometry
from modules.render.shaders import DrawRoi, MaskAA, MaskBlend, MaskBlur
from modules.utils.PointsAndRects import Rect, Point2f

# GL
from modules.gl import Fbo, SwapFbo, Texture


class CentreMaskLayer(LayerBase):
    """Renders mask image cropped and rotated with temporal blending and blur.

    Uses independent rotation calculation (aspect-corrected) compared to camera layer.
    Reads anchor points from AnchorPointCalculator.
    """

    def __init__(self, anchor_calc: CentreGeometry, cam_texture: Texture) -> None:
        self._anchor_calc: CentreGeometry = anchor_calc
        self._cam_texture: Texture = cam_texture

        # FBOs
        self._mask_fbo: Fbo = Fbo()
        self._mask_blend_fbo: SwapFbo = SwapFbo()
        self._mask_AA_fbo: Fbo = Fbo()
        self._mask_blur_fbo: SwapFbo = SwapFbo()

        # Shaders
        self._roi_shader = DrawRoi()
        self._mask_blend_shader = MaskBlend()
        self._mask_AA_shader = MaskAA()
        self._mask_blur_shader = MaskBlur()

        # Configuration
        self.blend_factor: float = 0.33
        self.blur_steps: int = 0
        self.blur_radius: float = 8.0

    @property
    def texture(self) -> Texture:
        """Output texture for external use."""
        return self._mask_blur_fbo.texture

    @property
    def mask_blur_fbo(self) -> SwapFbo:
        """Direct FBO access for compositor."""
        return self._mask_blur_fbo

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._mask_fbo.allocate(width, height, GL_R32F)
        self._mask_blend_fbo.allocate(width, height, GL_R32F)
        self._mask_AA_fbo.allocate(width, height, GL_R32F)
        self._mask_blur_fbo.allocate(width, height, GL_R32F)

        self._roi_shader.allocate()
        self._mask_blend_shader.allocate()
        self._mask_AA_shader.allocate()
        self._mask_blur_shader.allocate()

    def deallocate(self) -> None:
        self._mask_fbo.deallocate()
        self._mask_blend_fbo.deallocate()
        self._mask_AA_fbo.deallocate()
        self._mask_blur_fbo.deallocate()

        self._roi_shader.deallocate()
        self._mask_blend_shader.deallocate()
        self._mask_AA_shader.deallocate()
        self._mask_blur_shader.deallocate()

    def update(self) -> None:
        """Render mask crop using bbox geometry from CentreGeometry."""
        # Disable blending during FBO rendering
        glDisable(GL_BLEND)
        glColor4f(1.0, 1.0, 1.0, 1.0)

        self._mask_fbo.clear(0.0, 0.0, 0.0, 0.0)

        # Check if valid anchor data exists
        if not self._anchor_calc.has_pose:
            self._mask_blend_fbo.clear(0.0, 0.0, 0.0, 0.0)
            self._mask_blend_fbo.swap()
            self._mask_blend_fbo.clear(0.0, 0.0, 0.0, 0.0)
            glEnable(GL_BLEND)
            return

        # Use bbox geometry from CentreGeometry
        self._roi_shader.use(
            self._mask_fbo.fbo_id,
            self._cam_texture.tex_id,
            self._anchor_calc.bbox_crop_roi,
            self._anchor_calc.bbox_rotation,
            self._anchor_calc.bbox_rotation_center,
            self._anchor_calc.bbox_aspect,
            False,
            False
        )

        # Temporal blending
        self._mask_blend_fbo.swap()
        self._mask_blend_shader.use(
            self._mask_blend_fbo.fbo_id,
            self._mask_blend_fbo.back_tex_id,
            self._mask_fbo.tex_id,
            self.blend_factor
        )

        # Anti-aliasing
        self._mask_AA_shader.use(self._mask_AA_fbo.fbo_id, self._mask_blend_fbo.tex_id)

        # Copy AA result to blur buffer
        self._mask_blur_fbo.begin()
        self._mask_AA_fbo.draw(0, 0, self._mask_blur_fbo.width, self._mask_blur_fbo.height)
        self._mask_blur_fbo.end()

        # Multi-pass blur
        for i in range(self.blur_steps):
            self._mask_blur_fbo.swap()
            self._mask_blur_shader.use(
                self._mask_blur_fbo.fbo_id,
                self._mask_blur_fbo.back_tex_id,
                True,
                self.blur_radius
            )
            self._mask_blur_fbo.swap()
            self._mask_blur_shader.use(
                self._mask_blur_fbo.fbo_id,
                self._mask_blur_fbo.back_tex_id,
                False,
                self.blur_radius
            )

        glEnable(GL_BLEND)

