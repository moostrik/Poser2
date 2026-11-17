"""The color a bridge's density carries, made once a step for the 2D and the 3D bridge alike."""

from OpenGL.GL import GL_RGBA16F

from modules.gl import Fbo, Texture
from . import shaders
from .port import Port, TextureKind


class DensityColor:
    """The density's color: the image's RGB, or white without one, times the tint, gated by the mask's red,
    RGBA16F at the bridge's output size; its port `color`."""

    def __init__(self) -> None:
        self._color: Fbo = Fbo()
        self._shader: shaders.DensityColor = shaders.DensityColor()

    @property
    def color(self) -> Fbo:
        """The color the last update made."""
        return self._color

    @property
    def ports(self) -> dict[str, Port]:
        return {'color': Port(self._color, TextureKind.IMAGE)}

    def allocate(self, width: int, height: int) -> None:
        self._color.allocate(width, height, GL_RGBA16F)
        self._shader.allocate()

    def deallocate(self) -> None:
        self._color.deallocate()
        self._shader.deallocate()

    def update(self, image: Texture | None, tint: tuple[float, float, float], mask: Texture | None) -> None:
        """The color of `image`, or white without one, times `tint`, gated by `mask`, or ungated without one."""
        self._color.begin()
        self._shader.use(image, tint, mask)
        self._color.end()
