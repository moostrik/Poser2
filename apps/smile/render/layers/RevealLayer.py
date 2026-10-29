from modules.gl import Fbo, Image, SwapFbo, Texture, shaders

from .LayerBase import LayerBase
from .passes import blur, copy


class RevealLayer(LayerBase):
    """The flash out: an image appears from white, masked by the depth of the capture."""

    def __init__(self, depth: Image, white: Image) -> None:
        self._depth: Image = depth
        self._white: Image = white
        self._fbo: Fbo = Fbo()
        self._mask: SwapFbo = SwapFbo()
        self._flash_out: shaders.FlashOut = shaders.FlashOut()
        self._blur_h: shaders.BlurH = shaders.BlurH()
        self._blur_v: shaders.BlurV = shaders.BlurV()

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
        """flash 1 is all white, 0 is all image."""
        if not self._white.allocated or not image.allocated:
            return
        self._fbo.begin()
        self._flash_out.use(image, self._white, self._mask.texture, flash)
        self._fbo.end()
