"""ViewMask: where the player is in the view."""

# Third-party imports
from OpenGL.GL import GL_R16F

# Local application imports
from modules.flow import Port, TextureKind
from modules.gl import Fbo, Texture


class ViewMask:
    """The player's mask at the view's size, its own texture: full (1 everywhere) until the segmentation comes in."""

    def __init__(self) -> None:
        self._mask: Fbo = Fbo()

    @property
    def mask(self) -> Texture:
        return self._mask

    @property
    def ports(self) -> dict[str, Port]:
        return {'mask': Port(self._mask, TextureKind.SCALAR)}

    def allocate(self, width: int, height: int) -> None:
        self._mask.allocate(width, height, GL_R16F)
        self.fill()

    def deallocate(self) -> None:
        self._mask.deallocate()

    def clear(self) -> None:
        self.fill()

    def fill(self) -> None:
        """The full mask: the whole view is the player's."""
        self._mask.clear(1.0, 1.0, 1.0, 1.0)
