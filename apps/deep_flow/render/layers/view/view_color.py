"""ViewColor: the view's last pass, the player's color for the density: its image graded and sharpened, its alpha
a matte from its brightness, then premultiplied by the mask. As compositing does it: the corrections on the straight image, the matte applied after
them, eroded and feathered where the picture is made, so the mask the motion and the depth use stays as it is."""

# Third-party imports
from OpenGL.GL import GL_R16F

# Local application imports
from modules.flow import Port, TextureKind
from modules.gl import Fbo, SwapFbo, Style, Texture, shaders
from ....settings import ViewColorSettings
from .shaders import Feather, Grade


class ViewColor:
    """`process` grades an image (`Grade`) and sharpens it where `sharpen`, while `enabled`, sets its alpha by its
    brightness where `luma_matte` (`LumaMatte`), then premultiplies it by the mask where `use_mask` (RGB and alpha alike, as the render blends), into its own buffer; the mask's copy
    first eroded by `erode` pixels, then feathered inward over `feather` pixels. `texture` holds the result."""

    def __init__(self, settings: ViewColorSettings) -> None:
        self.settings: ViewColorSettings = settings
        self._fbo: SwapFbo = SwapFbo()
        self._matte: SwapFbo = SwapFbo()                    # the mask's copy, eroded and blurred in turn
        self._feathered: Fbo = Fbo()                        # the mask's copy feathered, the premultiply's
        self._grade_shader: Grade = Grade()
        self._sharpen_shader: shaders.Sharpen = shaders.Sharpen()
        self._luma_matte_shader: shaders.LumaMatte = shaders.LumaMatte()
        self._erode_shader: shaders.Erode = shaders.Erode()
        self._blur_shader: shaders.Blur = shaders.Blur()
        self._feather_shader: Feather = Feather()
        self._mask_shader: shaders.MaskMultiply = shaders.MaskMultiply()     # the premultiply: RGB and alpha by the mask

    @property
    def texture(self) -> Texture:
        return self._fbo        # the SwapFbo itself: always its newest image, safe to keep

    @property
    def ports(self) -> dict[str, Port]:
        return {'color': Port(self._fbo, TextureKind.IMAGE)}

    @property
    def made_with(self) -> tuple:
        """The settings the result is made with, so its owner makes it again when one changes."""
        s: ViewColorSettings = self.settings
        return (s.enabled, s.exposure, s.offset, s.gamma, s.contrast, s.saturation, s.sharpen, s.luma_matte, s.use_mask,
                s.erode, s.feather)

    def _shaders(self) -> tuple:
        return (self._grade_shader, self._sharpen_shader, self._luma_matte_shader, self._erode_shader, self._blur_shader, self._feather_shader,
                self._mask_shader)

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._fbo.allocate(width, height, internal_format)
        self._matte.allocate(width, height, GL_R16F)
        self._feathered.allocate(width, height, GL_R16F)
        for shader in self._shaders():
            shader.allocate()

    def deallocate(self) -> None:
        for fbo in (self._fbo, self._matte, self._feathered):
            fbo.deallocate()
        for shader in self._shaders():
            shader.deallocate()

    def clear(self) -> None:
        self._fbo.clear()

    def process(self, image: Texture, mask: Texture) -> None:
        s: ViewColorSettings = self.settings
        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)
        self._fbo.swap()
        self._fbo.begin()
        if s.enabled:
            self._grade_shader.use(image, s.exposure, s.offset, s.gamma, s.contrast, s.saturation)
        else:
            image.draw()
        self._fbo.end()
        if s.enabled and s.sharpen > 0.0:
            self._fbo.swap()
            self._fbo.begin()
            self._sharpen_shader.use(self._fbo.back, s.sharpen)
            self._fbo.end()
        if s.luma_matte > 0.0:
            self._fbo.swap()
            self._fbo.begin()
            self._luma_matte_shader.use(self._fbo.back, s.luma_matte)
            self._fbo.end()
        if s.use_mask:
            matte: Texture = self._matte_of(mask, int(s.erode), s.feather)
            self._fbo.swap()
            self._fbo.begin()
            self._mask_shader.use(self._fbo.back, matte)
            self._fbo.end()
        Style.pop_style()

    def _matte_of(self, mask: Texture, erode: int, feather: float) -> Texture:
        """`mask`'s copy, its edge moved inward by `erode` pixels (one 3 × 3 minimum each), then faded in inward over
        `feather` pixels (a Gaussian of sigma `feather` / 2, 2 × it − 1); the mask itself with neither."""
        made: bool = False                  # whether the matte holds a pass's result, read as its back after a swap

        def source() -> Texture:
            return self._matte.back if made else mask

        for _ in range(max(erode, 0)):
            self._matte.swap()
            self._matte.begin()
            self._erode_shader.use(source(), 1.0)
            self._matte.end()
            made = True
        if feather > 0.0:
            for horizontal in (True, False):
                self._matte.swap()
                self._matte.begin()
                self._blur_shader.use(source(), horizontal, feather / 2.0)
                self._matte.end()
                made = True
            self._feathered.begin()
            self._feather_shader.use(self._matte)
            self._feathered.end()
            return self._feathered
        return self._matte if made else mask
