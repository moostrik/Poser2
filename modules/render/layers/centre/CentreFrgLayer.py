"""Renders centered and rotated foreground with temporal blending and optional mask."""

# Local application imports
from modules.settings import Field, BaseSettings
from ..LayerBase import LayerBase
from .CentreGeometry import CentreGeometry
from ...shaders import DrawRoi, Blend, MaskApply, CelShade, HueShift
from modules.gl import Fbo, SwapFbo, Texture, Style
from modules.gl.shaders import Sharpen
from ...color_settings import ColorSettings


class CentreFrgSettings(BaseSettings):
    """Configuration for CentreFrgLayer foreground rendering."""
    blend_factor:   Field[float] = Field(0.2, min=0.0, max=1.0, description="Foreground temporal blending")
    exposure:       Field[float] = Field(1.2, min=0.1, max=3.0, description="Exposure multiplier")
    gamma:          Field[float] = Field(1.0, min=0.5, max=2.0, description="Gamma correction")
    offset:         Field[float] = Field(0.0, min=-0.5, max=0.5, description="Exposure offset")
    contrast:       Field[float] = Field(1.1, min=0.5, max=2.0, description="Contrast")
    saturation:     Field[float] = Field(2.0, min=0.5, max=2.0, description="Color saturation")
    # Cel shade
    levels:         Field[int]   = Field(9, min=2, max=12, description="Number of color bands")
    smoothness:     Field[float] = Field(0.33, min=0.0, max=0.5, description="Gradient between bands")
    # Hue shift
    colorize:       Field[float] = Field(0.96, min=0.0, max=1.0, description="Hue shift toward track color (0.0 = off)")
    # Post
    sharpen:        Field[float] = Field(0.0, min=0.0, max=2.0, description="Sharpen result (0.0 = off)")
    use_mask:       Field[bool]  = Field(True, description="Apply mask to foreground")


class CentreFrgLayer(LayerBase):
    """Renders foreground image cropped and rotated with temporal blending.

    Uses bbox_geometry for rendering. Optionally applies mask texture for compositing.
    """

    def __init__(self, cam_id: int, geometry: CentreGeometry, frg_texture: Texture, mask_texture: Texture, settings: CentreFrgSettings, color_settings: ColorSettings) -> None:
        self._cam_id: int = cam_id
        self._geometry: CentreGeometry = geometry
        self._frg_texture: Texture = frg_texture
        self._mask_texture: Texture = mask_texture
        self._color_settings: ColorSettings = color_settings

        # Configuration
        self.settings: CentreFrgSettings = settings

        # FBOs
        self._roi_fbo: Fbo = Fbo()
        self._blend_fbo: SwapFbo = SwapFbo()
        self._effect_fbo: SwapFbo = SwapFbo()

        # Shaders
        self._roi_shader = DrawRoi()
        self._blend_shader = Blend()
        self._cel_shade_shader = CelShade()
        self._hue_shift_shader = HueShift()
        self._sharpen_shader = Sharpen()
        self._mask_shader = MaskApply()

    @property
    def texture(self) -> Texture:
        """Output texture for external use."""
        return self._effect_fbo.texture

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._roi_fbo.allocate(width, height, internal_format)
        self._blend_fbo.allocate(width, height, internal_format)
        self._effect_fbo.allocate(width, height, internal_format)

        self._roi_shader.allocate()
        self._blend_shader.allocate()
        self._mask_shader.allocate()
        self._cel_shade_shader.allocate()
        self._hue_shift_shader.allocate()
        self._sharpen_shader.allocate()
        self._mask_shader.allocate()

    def deallocate(self) -> None:
        self._roi_fbo.deallocate()
        self._blend_fbo.deallocate()
        self._effect_fbo.deallocate()

        self._roi_shader.deallocate()
        self._blend_shader.deallocate()
        self._cel_shade_shader.deallocate()
        self._hue_shift_shader.deallocate()
        self._sharpen_shader.deallocate()
        self._mask_shader.deallocate()

    def update(self) -> None:
        """Render foreground crop using bbox geometry, with temporal blending and optional mask."""
        if self._geometry.lost:
            self._blend_fbo.clear(0.0, 0.0, 0.0, 0.0)
            self._blend_fbo.swap()
            self._blend_fbo.clear(0.0, 0.0, 0.0, 0.0)

        if self._geometry.crop_pose_points is None:
            return

        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)

        # Render foreground with bbox ROI
        self._roi_fbo.begin()
        self._roi_shader.use(
            self._frg_texture,
            self._geometry.bbox_geometry.crop_roi,
            self._geometry.bbox_geometry.rotation,
            self._geometry.bbox_geometry.rotation_center,
            self._geometry.bbox_geometry.aspect
        )
        self._roi_fbo.end()

        # Temporal blending
        self._blend_fbo.swap()
        self._blend_fbo.begin()
        self._blend_shader.use(
            self._blend_fbo.back_texture,
            self._roi_fbo.texture,
            self.settings.blend_factor
        )
        self._blend_fbo.end()

        self._cel_shade_shader.reload()
        self._effect_fbo.begin()
        self._cel_shade_shader.use(
            self._blend_fbo.texture,
            self.settings.exposure, self.settings.gamma, self.settings.offset,
            self.settings.contrast,
            self.settings.levels, self.settings.smoothness,
            self.settings.saturation
        )
        self._effect_fbo.end()

        # Hue shift toward track color
        if self.settings.colorize > 0.0 and self._color_settings is not None:
            self._effect_fbo.swap()
            self._effect_fbo.begin()
            r, g, b = self._color_settings.track_colors[self._cam_id % len(self._color_settings.track_colors)].rgb
            self._hue_shift_shader.use(self._effect_fbo.back_texture, r, g, b, self.settings.colorize)
            self._effect_fbo.end()

        # Sharpen
        if self.settings.sharpen > 0.0:
            self._effect_fbo.swap()
            self._effect_fbo.begin()
            self._sharpen_shader.use(self._effect_fbo.back_texture, self.settings.sharpen)
            self._effect_fbo.end()

        # Mask
        if self._mask_texture and self.settings.use_mask:
            self._effect_fbo.swap()
            self._effect_fbo.begin()
            self._mask_shader.use(self._effect_fbo.back_texture, self._mask_texture)
            self._effect_fbo.end()

        Style.pop_style()