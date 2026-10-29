from modules.gl import Image, SwapFbo, Texture, shaders

from ..shaders.blend import Blend
from .layer_base import LayerBase
from .passes import blur, copy


class DiffusionLayer(LayerBase):
    """The generated images, each blended over the previous result (the trail)."""

    def __init__(self, image: Image) -> None:
        self._image: Image = image
        self._fbo: SwapFbo = SwapFbo()
        self._blend: Blend = Blend()
        self._blur_h: shaders.BlurH = shaders.BlurH()
        self._blur_v: shaders.BlurV = shaders.BlurV()

    def allocate(self, width: int | None = None, height: int | None = None, internal_format: int | None = None) -> None:
        assert width and height and internal_format
        self._fbo.allocate(width, height, internal_format)
        self._blend.allocate()
        self._blur_h.allocate()
        self._blur_v.allocate()

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._blend.deallocate()
        self._blur_h.deallocate()
        self._blur_v.deallocate()

    def update(self) -> None:
        self._blend.reload()
        self._blur_h.reload()
        self._blur_v.reload()

    @property
    def texture(self) -> Texture:
        return self._fbo.texture

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
        self._blend.use(self._image, self._fbo.back_texture, trail)
        self._fbo.end()

    def blur(self, radius: float) -> None:
        """Blur the result in place; called every frame it adds up."""
        blur(self._fbo, self._blur_h, self._blur_v, radius)
