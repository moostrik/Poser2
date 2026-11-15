from modules.gl import Image, SwapFbo, Texture, shaders

from .layer_base import LayerBase
from .passes import blur, copy


class DiffusionLayer(LayerBase):
    """The generated images, each blended over the previous result (the trail)."""

    def __init__(self, image: Image) -> None:
        self._image: Image = image
        self._fbo: SwapFbo = SwapFbo()
        self._mix: shaders.Mix = shaders.Mix()
        self._blur: shaders.Blur = shaders.Blur()

    def allocate(self, width: int | None = None, height: int | None = None, internal_format: int | None = None) -> None:
        assert width and height and internal_format
        self._fbo.allocate(width, height, internal_format)
        self._mix.allocate()
        self._blur.allocate()

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._mix.deallocate()
        self._blur.deallocate()

    def update(self) -> None:
        self._mix.reload()
        self._blur.reload()

    @property
    def texture(self) -> Texture:
        return self._fbo

    def seed(self, texture: Texture) -> None:
        """Start the trail from this image (the capture)."""
        if not texture.allocated:
            return
        copy(texture, self._fbo)

    def render(self, trail: float) -> None:
        """Blend the newest image over the result; trail 0 shows only the newest."""
        if not self._image.allocated:
            return
        self._fbo.swap()
        self._fbo.begin()
        self._mix.use(self._image, self._fbo.back, trail)
        self._fbo.end()

    def blur(self, sigma: float) -> None:
        """Blur the result in place by `sigma` pixels; called every frame it adds up."""
        blur(self._fbo, self._blur, sigma)
