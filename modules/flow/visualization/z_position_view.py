"""A z position as a monitor draws it: the z position (0 front to 1 back) in red, its weight in green."""

from OpenGL.GL import GL_RG16F

from modules.gl import Fbo, Texture
from .shaders import ZPositionResolve


class ZPositionView:
    """Resolves a z position (red the z position times its weight, green the weight) for the views to draw."""

    def __init__(self) -> None:
        self._fbo: Fbo = Fbo()
        self._shader: ZPositionResolve = ZPositionResolve()

    def allocate(self) -> None:
        self._shader.allocate()

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._shader.deallocate()

    def process(self, z_position: Texture) -> Texture:
        """The z position in red and its weight in green (RG16F), at `z_position`'s size."""
        if not self._fbo.allocated or (self._fbo.width, self._fbo.height) != (z_position.width, z_position.height):
            self._fbo.allocate(z_position.width, z_position.height, GL_RG16F)
        self._fbo.begin()
        self._shader.use(z_position)
        self._fbo.end()
        return self._fbo
