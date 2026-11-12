"""The foreground's look: the player's colors blended over time, cel shaded, their hue shifted toward the camera's
track color, sharpened, and the mask as alpha. `ForegroundLook` is the chain, which `CenterForegroundLayer` runs on
its cut-out; `ForegroundLookLayer` runs it on a foreground another layer made."""

# Local application imports
from modules.gl import Fbo, SwapFbo, Texture, Style, shaders
from modules.settings import Field, BaseSettings
from ..layer_base import LayerBase, HasCaptureTime
from ...shaders import Blend, MaskApply, CelShade, HueShift
from ...color_settings import ColorSettings


class ForegroundLookSettings(BaseSettings):
    """The foreground's look."""
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


class ForegroundLook:
    """The chain, into its own buffers: `process` takes a foreground and its mask, `texture` holds the result,
    `clear` forgets what the temporal blend holds."""

    def __init__(self, cam_id: int, settings: ForegroundLookSettings, color_settings: ColorSettings) -> None:
        self._cam_id: int = cam_id
        self.settings: ForegroundLookSettings = settings
        self._color_settings: ColorSettings = color_settings

        self._blend_fbo: SwapFbo = SwapFbo()
        self._effect_fbo: SwapFbo = SwapFbo()

        self._blend_shader = Blend()
        self._cel_shade_shader = CelShade()
        self._hue_shift_shader = HueShift()
        self._sharpen_shader = shaders.Sharpen()
        self._mask_shader = MaskApply()

    @property
    def texture(self) -> Texture:
        return self._effect_fbo        # the SwapFbo itself: always its newest image, safe to keep

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._blend_fbo.allocate(width, height, internal_format)
        self._effect_fbo.allocate(width, height, internal_format)
        self._blend_shader.allocate()
        self._cel_shade_shader.allocate()
        self._hue_shift_shader.allocate()
        self._sharpen_shader.allocate()
        self._mask_shader.allocate()

    def deallocate(self) -> None:
        self._blend_fbo.deallocate()
        self._effect_fbo.deallocate()
        self._blend_shader.deallocate()
        self._cel_shade_shader.deallocate()
        self._hue_shift_shader.deallocate()
        self._sharpen_shader.deallocate()
        self._mask_shader.deallocate()

    def clear(self) -> None:
        self._blend_fbo.clear(0.0, 0.0, 0.0, 0.0)
        self._blend_fbo.swap()
        self._blend_fbo.clear(0.0, 0.0, 0.0, 0.0)

    def process(self, source: Texture, mask: Texture | None) -> None:
        """The look of `source`, the player's colors on black, with `mask` as its alpha where `use_mask`."""
        s: ForegroundLookSettings = self.settings
        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)

        self._blend_fbo.swap()
        self._blend_fbo.begin()
        self._blend_shader.use(self._blend_fbo.back, source, s.blend_factor)
        self._blend_fbo.end()

        self._cel_shade_shader.reload()
        self._effect_fbo.swap()
        self._effect_fbo.begin()
        self._cel_shade_shader.use(self._blend_fbo, s.exposure, s.gamma, s.offset, s.contrast, s.levels, s.smoothness, s.saturation)
        self._effect_fbo.end()

        if s.colorize > 0.0 and self._color_settings is not None:
            self._effect_fbo.swap()
            self._effect_fbo.begin()
            r, g, b = self._color_settings.track_colors[self._cam_id % len(self._color_settings.track_colors)].rgb
            self._hue_shift_shader.use(self._effect_fbo.back, r, g, b, s.colorize)
            self._effect_fbo.end()

        if s.sharpen > 0.0:
            self._effect_fbo.swap()
            self._effect_fbo.begin()
            self._sharpen_shader.use(self._effect_fbo.back, s.sharpen)
            self._effect_fbo.end()

        if mask is not None and mask.allocated and s.use_mask:
            self._effect_fbo.swap()
            self._effect_fbo.begin()
            self._mask_shader.use(self._effect_fbo.back, mask)
            self._effect_fbo.end()

        Style.pop_style()


class ForegroundLookLayer(LayerBase):
    """The look of `source`'s foreground, the player's colors on black, with `mask` as alpha. Runs again on each new
    frame of the source, told by its capture time; `timestamp` forwards it."""

    def __init__(self, cam_id: int, source: LayerBase, mask: Texture, settings: ForegroundLookSettings, color_settings: ColorSettings) -> None:
        self._source: LayerBase = source
        self._mask: Texture = mask
        self.settings: ForegroundLookSettings = settings
        self._look: ForegroundLook = ForegroundLook(cam_id, settings, color_settings)
        self._timestamp: float | None = None

    @property
    def texture(self) -> Texture:
        return self._look.texture

    @property
    def timestamp(self) -> float | None:
        """The capture time of the frame the look it holds was made from, its source's; None without one."""
        return self._timestamp

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._look.allocate(width, height, internal_format)

    def deallocate(self) -> None:
        self._look.deallocate()

    def update(self) -> None:
        timestamp: float | None = self._source.timestamp if isinstance(self._source, HasCaptureTime) else None
        if timestamp is None:
            if self._timestamp is not None:
                self._look.clear()
                self._timestamp = None
            return
        if timestamp == self._timestamp or not self._source.texture.allocated:
            return
        self._timestamp = timestamp
        self._look.process(self._source.texture, self._mask)
