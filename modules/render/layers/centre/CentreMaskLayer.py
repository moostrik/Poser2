"""Renders centered and rotated mask with temporal blending and blur."""

# Standard library imports
from dataclasses import dataclass

# Third-party imports
from OpenGL.GL import GL_R16F

# Local application imports
from modules.ConfigBase import ConfigBase, config_field
from modules.render.layers.LayerBase import LayerBase
from modules.render.layers.centre.CentreGeometry import CentreGeometry
from modules.render.shaders import DrawRoi, MaskAA, MaskBlend, MaskBlur, MaskDilate
from modules.gl import Fbo, SwapFbo, Texture, Style
from modules.utils import HotReloadMethods


@dataclass
class CentreMaskConfig(ConfigBase):
    """Configuration for CentreMaskLayer temporal blending and blur."""
    blend_factor: float = config_field(0.2, min=0.0, max=1.0, description="Temporal blending strength (0=static, 1=instant)")
    dilation_steps: int = config_field(0, min=0, max=5, description="Number of mask dilation iterations")
    blur_steps: int = config_field(0, min=0, max=10, description="Number of blur iterations")
    blur_radius: float = config_field(8.0, min=0.0, max=20.0, description="Blur kernel radius in pixels")


class CentreMaskLayer(LayerBase):
    """Renders mask image cropped and rotated with temporal blending and blur."""

    def __init__(self, geometry: CentreGeometry, cam_texture: Texture, config: CentreMaskConfig | None = None) -> None:
        self._geometry: CentreGeometry = geometry
        self._cam_texture: Texture = cam_texture

        # Configuration
        self.config: CentreMaskConfig = config or CentreMaskConfig()

        # FBOs
        self._roi_fbo: Fbo = Fbo()  # Single temp buffer for ROI
        self._blend_fbo: SwapFbo = SwapFbo()
        self._mask_blur_fbo: SwapFbo = SwapFbo()  # Used for AA output, dilation, and blur

        # Shaders
        self._roi_shader = DrawRoi()
        self._mask_blend_shader = MaskBlend()
        self._mask_AA_shader = MaskAA()
        self._mask_dilate_shader = MaskDilate()
        self._mask_blur_shader = MaskBlur()

        self.hot_reloader = HotReloadMethods(self.__class__, True, True)

    @property
    def texture(self) -> Texture:
        """Output texture for external use."""
        return self._mask_blur_fbo.texture

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._roi_fbo.allocate(width, height, GL_R16F)
        self._blend_fbo.allocate(width, height, GL_R16F)
        self._mask_blur_fbo.allocate(width, height, GL_R16F)

        self._roi_shader.allocate()
        self._mask_blend_shader.allocate()
        self._mask_AA_shader.allocate()
        self._mask_dilate_shader.allocate()
        self._mask_blur_shader.allocate()

    def deallocate(self) -> None:
        self._roi_fbo.deallocate()
        self._blend_fbo.deallocate()
        self._mask_blur_fbo.deallocate()

        self._roi_shader.deallocate()
        self._mask_blend_shader.deallocate()
        self._mask_AA_shader.deallocate()
        self._mask_dilate_shader.deallocate()
        self._mask_blur_shader.deallocate()

    def update(self) -> None:
        """Render mask crop using bbox geometry from CentreGeometry."""
        if self._geometry.lost:
            self._blend_fbo.clear_all()
            self._mask_blur_fbo.clear_all()
            return

        if self._geometry.crop_pose_points is None:
            return

        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)

        # Render ROI to temp buffer
        self._roi_fbo.begin()
        self._roi_shader.use(
            self._cam_texture,
            self._geometry.bbox_geometry.crop_roi,
            self._geometry.bbox_geometry.rotation,
            self._geometry.bbox_geometry.rotation_center,
            self._geometry.bbox_geometry.aspect
        )
        self._roi_fbo.end()

        # Temporal blending
        self._blend_fbo.swap()
        self._blend_fbo.begin()
        self._mask_blend_shader.use(
            self._blend_fbo.back_texture,
            self._roi_fbo.texture,
            self.config.blend_factor
        )
        self._blend_fbo.end()

        # AA directly to blur buffer
        self._mask_blur_fbo.begin()
        self._mask_AA_shader.use(self._blend_fbo.texture)
        self._mask_blur_fbo.end()

        # Multi-pass dilation
        for i in range(self.config.dilation_steps):
            self._mask_blur_fbo.swap()
            self._mask_blur_fbo.begin()
            self._mask_dilate_shader.use(self._mask_blur_fbo.back_texture, 1.0)
            self._mask_blur_fbo.end()

        # Multi-pass blur
        for i in range(self.config.blur_steps):
            self._mask_blur_fbo.swap()
            self._mask_blur_fbo.begin()
            self._mask_blur_shader.use(
                self._mask_blur_fbo.back_texture,
                True,
                self.config.blur_radius
            )
            self._mask_blur_fbo.end()

            self._mask_blur_fbo.swap()
            self._mask_blur_fbo.begin()
            self._mask_blur_shader.use(
                self._mask_blur_fbo.back_texture,
                False,
                self.config.blur_radius
            )
            self._mask_blur_fbo.end()

        Style.pop_style()
