"""DensityColor: the density's color, the `color` stage's work: the view's image graded and sharpened, its alpha a
matte from its brightness, then premultiplied by the mask. As compositing does it: the corrections on the straight
image, the matte applied after them, eroded and feathered where the picture is made, so the mask the motion and the
depth use stays as it is."""

import numpy as np
from OpenGL.GL import GL_R16F

from modules.gl import Fbo, SwapFbo, Style, Texture, shaders
from modules.settings import BaseSettings, Field
from ..port import Port, TextureKind
from .white_balance import white_balance_matrix


class DensityColorSettings(BaseSettings):
    """The image white balanced and graded, in the order applied, then sharpened, its alpha a matte from its
    brightness; then premultiplied by the mask's copy, eroded and feathered."""
    enabled:    Field[bool]  = Field(True, description="The color graded; off, the image as it is")
    white_balance: Field[float] = Field(6500.0, min=2000.0, max=12000.0, step=50.0, newline=True, description="The light the color is corrected for (K); lower cools, higher warms; 6500 = unchanged")
    exposure:   Field[float] = Field(1.0, min=0.1, max=3.0, description="The color times this")
    offset:     Field[float] = Field(0.0, min=-0.5, max=0.5, description="Added to the color, after the exposure")
    gamma:      Field[float] = Field(1.0, min=0.5, max=2.0, description="Gamma")
    contrast:   Field[float] = Field(1.0, min=0.5, max=2.0, description="Contrast around mid grey")
    saturation: Field[float] = Field(1.0, min=0.0, max=2.0, description="Saturation, times this")
    sharpen:    Field[float] = Field(0.0, min=0.0, max=2.0, description="Sharpen (0 = off)")
    luma_matte: Field[float] = Field(1.0, min=0.0, max=1.0, description="Alpha from the color's brightness: darkness made transparency (0 = off)")
    use_mask:  Field[bool]  = Field(True, newline=True, description="The color cut by the mask, premultiplied")
    erode:      Field[int]   = Field(0, min=0, max=4, step=1, description="Pixels the color's edge moves inside the mask's (view pixels)")
    feather:    Field[float] = Field(0.0, min=0.0, max=16.0, step=0.5, description="Width the color fades in over inside its edge (view pixels); 0 = hard")


class DensityColor:
    """`process` white balances and grades an image (`Grade`) and sharpens it where `sharpen`, while `enabled`,
    sets its alpha by its brightness where `luma_matte` (`LumaMatte`), then premultiplies it by the mask where
    `use_mask` (RGB and alpha alike, as the render blends), into its own buffer; the mask's copy first eroded by
    `erode` pixels, then feathered inward over `feather` pixels. `texture` holds the result."""

    def __init__(self, settings: DensityColorSettings) -> None:
        self.settings: DensityColorSettings = settings
        self._fbo: SwapFbo = SwapFbo()
        self._matte: SwapFbo = SwapFbo()                    # the mask's copy, eroded and blurred in turn
        self._feathered: Fbo = Fbo()                        # the mask's copy feathered, the premultiply's
        self._grade_shader: shaders.Grade = shaders.Grade()
        self._white_balance: tuple[float, np.ndarray] | None = None     # the kelvin and its matrix, made when it changes
        self._sharpen_shader: shaders.Sharpen = shaders.Sharpen()
        self._luma_matte_shader: shaders.LumaMatte = shaders.LumaMatte()
        self._erode_shader: shaders.Erode = shaders.Erode()
        self._blur_shader: shaders.Blur = shaders.Blur()
        self._feather_shader: shaders.Feather = shaders.Feather()
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
        s: DensityColorSettings = self.settings
        return (s.enabled, s.white_balance, s.exposure, s.offset, s.gamma, s.contrast, s.saturation, s.sharpen,
                s.luma_matte, s.use_mask, s.erode, s.feather)

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
        s: DensityColorSettings = self.settings
        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)
        self._fbo.swap()
        self._fbo.begin()
        if s.enabled:
            self._grade_shader.use(image, self._white_balance_matrix(s.white_balance), s.exposure, s.offset, s.gamma,
                                   s.contrast, s.saturation)
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

    def _white_balance_matrix(self, kelvin: float) -> np.ndarray:
        """The white balance's matrix for `kelvin`, made again only when it changes."""
        if self._white_balance is None or self._white_balance[0] != kelvin:
            self._white_balance = (kelvin, white_balance_matrix(kelvin))
        return self._white_balance[1]

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
