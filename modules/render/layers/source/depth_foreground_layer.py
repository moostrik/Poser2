# Third-party imports
from OpenGL.GL import GL_RGBA16F

# Local application imports
from modules.gl import Fbo, Style, Texture
from ..layer_base import LayerBase
from .capture_image_layer import CaptureImageLayer
from .depth_mask_layer import DepthMaskLayer
from ...shaders import MaskMultiply


class DepthForegroundLayer(LayerBase):
    """The capture's color image times the depth mask, premultiplied: the player's colors on black,
    at the image's size. The image and the depth are of one capture, so the mask fits the image.
    New with each capture; `CaptureImageLayer` and `DepthMaskLayer` update first."""

    def __init__(self, image: CaptureImageLayer, mask: DepthMaskLayer) -> None:
        self._image: CaptureImageLayer = image
        self._mask: DepthMaskLayer = mask
        self._fbo: Fbo = Fbo()
        self._shader: MaskMultiply = MaskMultiply()
        self._dirty: bool = False

    @property
    def texture(self) -> Texture:
        return self._fbo

    @property
    def dirty(self) -> bool:
        return self._dirty

    def allocate(self, width: int | None = None, height: int | None = None, internal_format: int | None = None) -> None:
        self._shader.allocate()     # the FBO on the first image, at its size

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._shader.deallocate()

    def update(self) -> None:
        self._dirty = False
        image: Texture = self._image.texture
        if not self._image.dirty or not self._mask.texture.allocated:
            return
        if not self._fbo.allocated:
            self._fbo.allocate(image.width, image.height, GL_RGBA16F)

        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)
        self._fbo.begin()
        self._shader.use(image, self._mask.texture)
        self._fbo.end()
        Style.pop_style()
        self._dirty = True
