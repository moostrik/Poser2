from modules.gl import Fbo, Image, SwapFbo, Texture
from modules.gl.shaders.convenience.BlurH import BlurH
from modules.gl.shaders.convenience.BlurV import BlurV
from modules.gl.shaders.convenience.FlashOut import FlashOut

from .LayerBase import LayerBase
from .passes import blur, copy


class RevealLayer(LayerBase):
    """The flash out: an image appears from the logo (white), masked by the depth of the capture."""

    def __init__(self, depth: Image, logo: Image) -> None:
        self._depth: Image = depth
        self._logo: Image = logo
        self._fbo: Fbo = Fbo()
        self._mask: SwapFbo = SwapFbo()
        self._flash_out: FlashOut = FlashOut()
        self._blur_h: BlurH = BlurH()
        self._blur_v: BlurV = BlurV()

    def allocate(self, width: int | None = None, height: int | None = None, internal_format: int | None = None) -> None:
        assert width and height and internal_format
        self._fbo.allocate(width, height, internal_format)
        self._mask.allocate(width, height, internal_format)
        self._flash_out.allocate()
        self._blur_h.allocate()
        self._blur_v.allocate()

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._mask.deallocate()
        self._flash_out.deallocate()
        self._blur_h.deallocate()
        self._blur_v.deallocate()

    def update(self) -> None:
        self._flash_out.reload()
        self._blur_h.reload()
        self._blur_v.reload()

    @property
    def texture(self) -> Texture:
        return self._fbo

    def set_mask(self) -> None:
        """Take the newest capture depth as the mask, softened."""
        if not self._depth.allocated:
            return
        copy(self._depth, self._mask)
        blur(self._mask, self._blur_h, self._blur_v, 1.0)

    def render(self, image: Texture, flash: float) -> None:
        """flash 1 is all logo, 0 is all image."""
        if not self._logo.allocated or not image.allocated:
            return
        self._fbo.begin()
        self._flash_out.use(image, self._logo, self._mask.texture, flash)
        self._fbo.end()
