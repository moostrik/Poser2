"""Renders centered and rotated foreground with temporal blending and optional mask."""

# Local application imports
from modules.settings import Field, BaseSettings
from ..layer_base import LayerBase, HasCaptureTime
from .center_geometry import CenterGeometry, SourceSpace
from ...shaders import DrawRoi, Blend, MaskApply, MaskMultiply, CelShade, HueShift
from modules.gl import Fbo, SwapFbo, Texture, Style, shaders
from ...color_settings import ColorSettings


from modules.utils import HotReloadMethods

class CenterForegroundLayerSettings(BaseSettings):
    """Configuration for CenterForegroundLayer foreground rendering."""
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


class CenterForegroundLayer(LayerBase):
    """Renders foreground image cropped and rotated with temporal blending.

    `space` says whether the foreground is on the pose's crop or on the camera image. With
    `premultiply` the source is a whole image, and the mask multiplies it right after the cut-out, so
    the look starts from the player's colors on black as from a masked source (RVM's is). Optionally
    applies mask texture as alpha for compositing. `source` is the layer whose texture it draws;
    `timestamp` is that frame's capture time, forwarded.
    """

    def __init__(self, cam_id: int, geometry: CenterGeometry, source: LayerBase, mask_texture: Texture, space: SourceSpace,
                 premultiply: bool, settings: CenterForegroundLayerSettings, color_settings: ColorSettings) -> None:
        self._cam_id: int = cam_id
        self._geometry: CenterGeometry = geometry
        self._source: LayerBase = source
        self._timestamp: float | None = None     # the capture time of the frame drawn last
        self._space: SourceSpace = space
        self._premultiply: bool = premultiply
        self._mask_texture: Texture = mask_texture
        self._color_settings: ColorSettings = color_settings

        # Configuration
        self.settings: CenterForegroundLayerSettings = settings

        # FBOs
        self._roi_fbo: Fbo = Fbo()
        self._premultiplied_fbo: Fbo = Fbo()    # the cut-out times the mask, with `premultiply`
        self._blend_fbo: SwapFbo = SwapFbo()
        self._effect_fbo: SwapFbo = SwapFbo()

        # Shaders
        self._roi_shader = DrawRoi()
        self._blend_shader = Blend()
        self._cel_shade_shader = CelShade()
        self._hue_shift_shader = HueShift()
        self._sharpen_shader = shaders.Sharpen()
        self._mask_shader = MaskApply()
        self._premultiply_shader = MaskMultiply()

        self._hot_reload = HotReloadMethods(self.__class__, True, True)

    @property
    def texture(self) -> Texture:
        """Output texture for external use."""
        return self._effect_fbo        # the SwapFbo itself: always its newest image, safe to keep

    @property
    def timestamp(self) -> float | None:
        """The capture time of the camera frame the foreground it holds was made from, its source's; None without one."""
        return self._timestamp

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._roi_fbo.allocate(width, height, internal_format)
        if self._premultiply:
            self._premultiplied_fbo.allocate(width, height, internal_format)
            self._premultiply_shader.allocate()
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
        self._premultiplied_fbo.deallocate()
        self._premultiply_shader.deallocate()
        self._blend_fbo.deallocate()
        self._effect_fbo.deallocate()

        self._roi_shader.deallocate()
        self._blend_shader.deallocate()
        self._cel_shade_shader.deallocate()
        self._hue_shift_shader.deallocate()
        self._sharpen_shader.deallocate()
        self._mask_shader.deallocate()

    def update(self) -> None:
        """Render the foreground centered by CenterGeometry in its space, with temporal blending and optional mask."""
        if self._geometry.lost:
            self._blend_fbo.clear(0.0, 0.0, 0.0, 0.0)
            self._blend_fbo.swap()
            self._blend_fbo.clear(0.0, 0.0, 0.0, 0.0)
            self._timestamp = None

        source: Texture = self._source.texture
        if not self._geometry.ready or not source.allocated:
            return
        self._timestamp = self._source.timestamp if isinstance(self._source, HasCaptureTime) else None

        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)

        # Render foreground with its ROI
        self._roi_fbo.begin()
        self._roi_shader.use(source, *self._geometry.source_roi(self._space, source))
        self._roi_fbo.end()

        # A whole image: the player's colors on black, as a masked source arrives
        source: Texture = self._roi_fbo
        if self._premultiply and self._mask_texture.allocated:
            self._premultiplied_fbo.begin()
            self._premultiply_shader.use(self._roi_fbo, self._mask_texture)
            self._premultiplied_fbo.end()
            source = self._premultiplied_fbo

        # Temporal blending
        self._blend_fbo.swap()
        self._blend_fbo.begin()
        self._blend_shader.use(
            self._blend_fbo.back,
            source,
            self.settings.blend_factor
        )
        self._blend_fbo.end()

        self._cel_shade_shader.reload()
        self._effect_fbo.swap()
        self._effect_fbo.begin()
        self._cel_shade_shader.use(
            self._blend_fbo,
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
            self._hue_shift_shader.use(self._effect_fbo.back, r, g, b, self.settings.colorize)
            self._effect_fbo.end()

        # Sharpen
        if self.settings.sharpen > 0.0:
            self._effect_fbo.swap()
            self._effect_fbo.begin()
            self._sharpen_shader.use(self._effect_fbo.back, self.settings.sharpen)
            self._effect_fbo.end()

        # Mask
        if self._mask_texture and self.settings.use_mask:
            self._effect_fbo.swap()
            self._effect_fbo.begin()
            self._mask_shader.use(self._effect_fbo.back, self._mask_texture)
            self._effect_fbo.end()

        Style.pop_style()