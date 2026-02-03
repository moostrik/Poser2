"""Renders centered and rotated mask with temporal blending and blur."""

# Third-party imports
from OpenGL.GL import GL_R16F

# Local application imports
from modules.render.layers.LayerBase import LayerBase
from modules.render.layers.source import MaskSourceLayer
from modules.render.layers.centre.CentreGeometry import CentreGeometry
from modules.render.shaders import DrawRoi, MaskAA, MaskBlend, MaskBlur

# GL
from modules.gl import Fbo, SwapFbo, Texture, Blit, Style

from modules.utils import HotReloadMethods


class CentreMaskLayer(LayerBase):
    """Renders mask image cropped and rotated with temporal blending and blur.

    Uses independent rotation calculation (aspect-corrected) compared to camera layer.
    Reads anchor points from AnchorPointCalculator.
    """

    def __init__(self, geometry: CentreGeometry, cam_texture: Texture) -> None:
        self._geometry: CentreGeometry = geometry
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

        self.hot_reloader = HotReloadMethods(self.__class__, True, True)

    @property
    def texture(self) -> Texture:
        """Output texture for external use."""
        return self._mask_blur_fbo.texture

    @property
    def mask_blur_fbo(self) -> SwapFbo:
        """Direct FBO access for compositor."""
        return self._mask_blur_fbo

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._mask_fbo.allocate(width, height, GL_R16F)
        self._mask_blend_fbo.allocate(width, height, GL_R16F)
        self._mask_AA_fbo.allocate(width, height, GL_R16F)
        self._mask_blur_fbo.allocate(width, height, GL_R16F)

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


        self.blend_factor: float = 0.2
        # print("CentreMaskLayer update - blend_factor set to", self.blend_factor)

        if self._geometry.lost:
            self._mask_fbo.clear()
            self._mask_blend_fbo.clear_all()

        # Check if valid anchor data exists
        if self._geometry.idle or self._geometry.empty:
            return

        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)

        # Use bbox geometry from CentreGeometry
        self._mask_fbo.begin()
        self._roi_shader.use(
            self._cam_texture,
            self._geometry.bbox_crop_roi,
            self._geometry.bbox_rotation,
            self._geometry.bbox_rotation_center,
            self._geometry.bbox_aspect
        )
        self._mask_fbo.end()

        # Temporal blending
        self._mask_blend_fbo.swap()
        self._mask_blend_fbo.begin()
        self._mask_blend_shader.use(
            self._mask_blend_fbo.back_texture,
            self._mask_fbo.texture,
            self.blend_factor
        )
        self._mask_blend_fbo.end()

        # Anti-aliasing
        self._mask_AA_fbo.begin()
        self._mask_AA_shader.use(self._mask_blend_fbo.texture)
        self._mask_AA_fbo.end()

        # Copy AA result to blur buffer
        self._mask_blur_fbo.begin()
        Blit.use(self._mask_AA_fbo.texture)
        self._mask_blur_fbo.end()

        # Multi-pass blur
        for i in range(self.blur_steps):
            self._mask_blur_fbo.swap()
            self._mask_blur_fbo.begin()
            self._mask_blur_shader.use(
                self._mask_blur_fbo.back_texture,
                True,
                self.blur_radius
            )
            self._mask_blur_fbo.end()

            self._mask_blur_fbo.swap()
            self._mask_blur_fbo.begin()
            self._mask_blur_shader.use(
                self._mask_blur_fbo.back_texture,
                False,
                self.blur_radius
            )
            self._mask_blur_fbo.end()

        Style.pop_style()
