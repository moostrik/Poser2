from modules.gl import Fbo, Image, SwapFbo, Texture, shaders

from ..shaders import Omission
from modules.render import LayerBase
from .passes import blur, copy

MASK_SIGMA: float = 9.0     # the mask's softening (pixels)


class BloomLayer(LayerBase):
    """The last image blooms and dissolves into the camera, masked by the depth of that image."""

    def __init__(self, depth: Image) -> None:
        self._depth: Image = depth
        self._fbo: Fbo = Fbo()
        self._mask: SwapFbo = SwapFbo()
        self._omission: Omission = Omission()
        self._blur: shaders.Blur = shaders.Blur()

    def allocate(self, width: int | None = None, height: int | None = None, internal_format: int | None = None) -> None:
        assert width and height and internal_format
        self._fbo.allocate(width, height, internal_format)
        self._mask.allocate(width, height, internal_format)
        self._omission.allocate()
        self._blur.allocate()

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._mask.deallocate()
        self._omission.deallocate()
        self._blur.deallocate()

    def update(self) -> None:
        """Nothing to advance per frame."""

    @property
    def texture(self) -> Texture:
        return self._fbo

    def set_mask(self) -> None:
        """Take the newest depth of the last image as the mask, softened."""
        if not self._depth.allocated:
            return
        copy(self._depth, self._mask)
        blur(self._mask, self._blur, MASK_SIGMA)

    def render(self, image: Texture, camera: Texture, bloom: float, omission: float) -> None:
        """bloom brightens the image, omission 0..1 dissolves it into the camera."""
        if not image.allocated or not camera.allocated:
            return
        self._fbo.begin()
        self._omission.use(image, camera, self._mask, bloom, omission)
        self._fbo.end()
