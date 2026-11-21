"""Colorize, hd_trio's copy: the player's colors blended over time, cel shaded, their hue shifted toward the
camera's track color, sharpened, and the mask as alpha; `CenterForegroundLayer` runs it on its cut-out."""

# Local application imports
from modules.gl import SwapFbo, Texture, BlendState, shaders
from modules.render import ColorSettings
from modules.utils import Color
from ....settings import CenterForegroundLayerSettings
from .shaders import CelShade


class Colorize:
    """The chain, into its own buffers: `process` takes a foreground and its mask, `texture` holds the result,
    `clear` forgets what the temporal blend holds."""

    def __init__(self, cam_id: int, settings: CenterForegroundLayerSettings, color_settings: ColorSettings) -> None:
        self._cam_id: int = cam_id
        self.settings: CenterForegroundLayerSettings = settings
        self._color_settings: ColorSettings = color_settings

        self._blend_fbo: SwapFbo = SwapFbo()
        self._effect_fbo: SwapFbo = SwapFbo()

        self._mix_shader = shaders.Mix()
        self._cel_shade_shader = CelShade()
        self._colorize_shader = shaders.Colorize()
        self._sharpen_shader = shaders.Sharpen()
        self._mask_shader = shaders.MaskApply()

    @property
    def texture(self) -> Texture:
        return self._effect_fbo        # the SwapFbo itself: always its newest image, safe to keep

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._blend_fbo.allocate(width, height, internal_format)
        self._effect_fbo.allocate(width, height, internal_format)
        self._mix_shader.allocate()
        self._cel_shade_shader.allocate()
        self._colorize_shader.allocate()
        self._sharpen_shader.allocate()
        self._mask_shader.allocate()

    def deallocate(self) -> None:
        self._blend_fbo.deallocate()
        self._effect_fbo.deallocate()
        self._mix_shader.deallocate()
        self._cel_shade_shader.deallocate()
        self._colorize_shader.deallocate()
        self._sharpen_shader.deallocate()
        self._mask_shader.deallocate()

    def clear(self) -> None:
        self._blend_fbo.clear(0.0, 0.0, 0.0, 0.0)
        self._blend_fbo.swap()
        self._blend_fbo.clear(0.0, 0.0, 0.0, 0.0)

    def process(self, source: Texture, mask: Texture | None) -> None:
        """`source` colorized, the player's colors on black, with `mask` as its alpha where `use_mask`; `source` as
        it is while not `enabled`."""
        s: CenterForegroundLayerSettings = self.settings
        BlendState.push()
        BlendState.set_blend_mode(BlendState.BlendMode.DISABLED)

        if not s.enabled:
            self._effect_fbo.swap()
            self._effect_fbo.begin()
            source.draw()
            self._effect_fbo.end()
            BlendState.pop()
            return

        self._blend_fbo.swap()
        self._blend_fbo.begin()
        self._mix_shader.use(self._blend_fbo.back, source, s.blend_factor)
        self._blend_fbo.end()

        self._effect_fbo.swap()
        self._effect_fbo.begin()
        self._cel_shade_shader.use(self._blend_fbo, s.exposure, s.gamma, s.offset, s.contrast, s.levels, s.smoothness, s.saturation)
        self._effect_fbo.end()

        if s.colorize > 0.0 and self._color_settings is not None:
            self._effect_fbo.swap()
            self._effect_fbo.begin()
            color: Color = self._color_settings.track_colors[self._cam_id % len(self._color_settings.track_colors)]
            self._colorize_shader.use(self._effect_fbo.back, color, s.colorize)
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

        BlendState.pop()
