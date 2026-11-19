"""ViewCrop: the window the view cuts from the camera image, at the view's size."""

# Third-party imports
from OpenGL.GL import GL_RGBA16F

# Local application imports
from modules.flow import Port, TextureKind
from modules.gl import Fbo, Texture
from modules.utils import Rect
from ....settings import ViewCropSettings


class ViewCrop:
    """`crop` writes the image inside the window; its own texture."""

    def __init__(self, settings: ViewCropSettings) -> None:
        self.settings: ViewCropSettings = settings
        self._image: Fbo = Fbo()

    @property
    def image(self) -> Texture:
        return self._image

    @property
    def ports(self) -> dict[str, Port]:
        return {'image': Port(self._image, TextureKind.IMAGE)}

    @property
    def made_with(self) -> tuple:
        """The settings its texture is made with, so its owner makes it again when one changes."""
        s: ViewCropSettings = self.settings
        return (s.center_x, s.center_y, s.height)

    def allocate(self, width: int, height: int) -> None:
        self._image.allocate(width, height, GL_RGBA16F)

    def deallocate(self) -> None:
        self._image.deallocate()

    def clear(self) -> None:
        self._image.clear()

    def crop(self, image: Texture, roi: Rect) -> None:
        """`image` across `roi`, its window on the camera image (uv, rows down)."""
        self._image.begin()
        image.draw(src=roi)
        self._image.end()
