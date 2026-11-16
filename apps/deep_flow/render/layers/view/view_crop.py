"""ViewCrop: the box the view crops in 3D. Its sides take the camera image's window at the view's size; its front and
back take the camera's depth inside it on the depth's own pixels, who."""

# Third-party imports
from OpenGL.GL import GL_NEAREST, GL_R32F, GL_RGBA16F

# Local application imports
from modules.flow import DepthCrop, Port, TextureKind
from modules.gl import Fbo, Texture
from modules.utils import Rect
from ....settings import ViewCropSettings


class ViewCrop:
    """`crop` writes the image inside the box's sides and the depth inside the box, 0 outside; both its own textures."""

    def __init__(self, settings: ViewCropSettings) -> None:
        self.settings: ViewCropSettings = settings
        self._image: Fbo = Fbo()
        self._depth: Fbo = Fbo(interpolation=GL_NEAREST)  # the depth's own pixels, never a distance in between
        self._depth_shader: DepthCrop = DepthCrop()

    @property
    def image(self) -> Texture:
        return self._image

    @property
    def depth(self) -> Texture:
        return self._depth

    @property
    def ports(self) -> dict[str, Port]:
        s: ViewCropSettings = self.settings
        return {'image': Port(self._image, TextureKind.IMAGE),
                'depth': Port(self._depth, TextureKind.DEPTH, depth_range=(s.near, s.far))}

    @property
    def made_with(self) -> tuple:
        """The settings its textures are made with, so its owner makes them again when one changes."""
        s: ViewCropSettings = self.settings
        return (s.center_x, s.center_y, s.height, s.near, s.far)

    def allocate(self, width: int, height: int) -> None:
        self._image.allocate(width, height, GL_RGBA16F)
        self._depth.allocate(width, height, GL_R32F)
        self._depth_shader.allocate()

    def deallocate(self) -> None:
        self._image.deallocate()
        self._depth.deallocate()
        self._depth_shader.deallocate()

    def clear(self) -> None:
        self._image.clear()
        self._depth.clear()

    def crop(self, image: Texture, depth: Texture | None, roi: Rect) -> None:
        """`image` across `roi`, its window on the camera image (uv, rows down), and `depth` (metres on the camera
        image, 0 for none) inside the box; no depth without one."""
        self._image.begin()
        image.draw(src=roi)
        self._image.end()
        self._depth.clear()
        if depth is None:
            return
        self._depth.begin()
        self._depth_shader.use(depth, roi, self.settings.near, self.settings.far)
        self._depth.end()
