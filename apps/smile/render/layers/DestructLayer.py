from modules.gl import Fbo, Image, SwapFbo, Texture, shaders

from .LayerBase import LayerBase
from .passes import blur, copy


class DestructLayer(LayerBase):
    """The last image blooms and dissolves into the camera, masked by the depth of that image."""

    def __init__(self, depth: Image) -> None:
        self._depth: Image = depth
        self._fbo: Fbo = Fbo()
        self._mask: SwapFbo = SwapFbo()
        self._omission: shaders.Omission = shaders.Omission()
        self._blur_h: shaders.BlurH = shaders.BlurH()
        self._blur_v: shaders.BlurV = shaders.BlurV()

    def allocate(self, width: int | None = None, height: int | None = None, internal_format: int | None = None) -> None:
        assert width and height and internal_format
        self._fbo.allocate(width, height, internal_format)
        self._mask.allocate(width, height, internal_format)
        self._omission.allocate()
        self._blur_h.allocate()
        self._blur_v.allocate()

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._mask.deallocate()
        self._omission.deallocate()
        self._blur_h.deallocate()
        self._blur_v.deallocate()

    def update(self) -> None:
        self._omission.reload()
        self._blur_h.reload()
        self._blur_v.reload()

    @property
    def texture(self) -> Texture:
        return self._fbo

    def set_mask(self) -> None:
        """Take the newest depth of the last image as the mask, softened."""
        if not self._depth.allocated:
            return
        copy(self._depth, self._mask)
        blur(self._mask, self._blur_h, self._blur_v, 1.0)

    def render(self, image: Texture, camera: Texture, bloom: float, omission: float) -> None:
        """bloom brightens the image, omission 0..1 dissolves it into the camera."""
        if not image.allocated or not camera.allocated:
            return
        self._fbo.begin()
        self._omission.use(image, camera, self._mask.texture, bloom, omission)
        self._fbo.end()
