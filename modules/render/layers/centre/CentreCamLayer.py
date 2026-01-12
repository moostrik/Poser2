"""Renders centered and rotated camera view based on pose anchor points with optional mask."""

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.render.layers.LayerBase import LayerBase
from modules.render.layers.centre.CentreGeometry import CentreGeometry
from modules.render.shaders import Blend, DrawRoi, MaskApply
from modules.utils.PointsAndRects import Rect

# GL
from modules.gl import Fbo, SwapFbo, Texture


class CentreCamLayer(LayerBase):
    """Renders camera image cropped and rotated around pose anchor points.

    Reads anchor geometry from CentreGeometry and applies DrawRoi shader
    followed by temporal blending. Optionally applies mask texture for compositing.
    """

    def __init__(self, anchor_calc: CentreGeometry, cam_texture: Texture, mask_texture: Texture | None = None, mask_opacity: float = 1.0) -> None:
        self._anchor_calc: CentreGeometry = anchor_calc
        self._cam_texture: Texture = cam_texture
        self._mask_texture: Texture | None = mask_texture

        # FBOs
        self._cam_fbo: Fbo = Fbo()
        self._cam_blend_fbo: SwapFbo = SwapFbo()
        self._masked_fbo: Fbo = Fbo()
        self._output_fbo: Fbo | SwapFbo = self._masked_fbo if self._mask_texture else self._cam_blend_fbo

        # Shaders
        self._roi_shader = DrawRoi()
        self._blend_shader = Blend()
        self._mask_shader = MaskApply()

        # Configuration
        self.blend_factor: float = 0.5
        self.mask_opacity: float = mask_opacity
        self.use_mask: bool = True  # Toggle for mask application

    @property
    def texture(self) -> Texture:
        """Output texture for external use."""
        return self._output_fbo.texture

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._cam_fbo.allocate(width, height, internal_format)
        self._cam_blend_fbo.allocate(width, height, internal_format)
        if self._mask_texture:
            self._masked_fbo.allocate(width, height, internal_format)
        self._roi_shader.allocate()
        self._blend_shader.allocate()
        self._mask_shader.allocate()

    def deallocate(self) -> None:
        self._cam_fbo.deallocate()
        self._cam_blend_fbo.deallocate()
        self._masked_fbo.deallocate()
        self._roi_shader.deallocate()
        self._blend_shader.deallocate()
        self._mask_shader.deallocate()

    def update(self) -> None:
        """Render camera crop using anchor geometry, optionally with mask."""
        # Disable blending during FBO rendering
        glDisable(GL_BLEND)
        glColor4f(1.0, 1.0, 1.0, 1.0)

        self._cam_fbo.clear(0.0, 0.0, 0.0, 0.0)

        # Check if valid anchor data exists
        if not self._anchor_calc.has_pose:
            self._cam_blend_fbo.clear(0.0, 0.0, 0.0, 0.0)
            self._cam_blend_fbo.swap()
            self._cam_blend_fbo.clear(0.0, 0.0, 0.0, 0.0)
            if self._mask_texture and self.use_mask:
                self._masked_fbo.clear(0.0, 0.0, 0.0, 0.0)
            glEnable(GL_BLEND)
            return

        # Render camera image with ROI from anchor calculator
        cam_aspect: float = self._cam_texture.width / self._cam_texture.height
        self._roi_shader.use(
            self._cam_fbo,
            self._cam_texture,
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
            self._cam_blend_fbo,
            self._cam_blend_fbo.back_texture,
            self._cam_fbo.texture,
            self.blend_factor
        )

        # Apply mask if provided and enabled
        if self._mask_texture and self.use_mask:
            self._masked_fbo.clear(0.0, 0.0, 0.0, 0.0)
            self._mask_shader.use(
                self._masked_fbo,
                self._cam_blend_fbo.texture,
                self._mask_texture,
                self.mask_opacity
            )
            self._output_fbo = self._masked_fbo
        else:
            self._output_fbo = self._cam_blend_fbo

        glEnable(GL_BLEND)
