"""ViewDepth: the camera's depth read into the view, within a surface or on the mask's side of an edge."""

# Third-party imports
from OpenGL.GL import GL_NEAREST, GL_R32F

# Local application imports
from modules.flow import DepthRead, DepthUpscale, Port, TextureKind
from modules.gl import Fbo, Texture
from modules.utils import Rect
from ....settings import ViewDepthMaskSettings, ViewDepthSettings
from .view_depth_mask import ViewDepthMask


class ViewDepth:
    """`read` brings a depth into the view, cut to the crop: `SURFACE` blends only the depth pixels within
    `tolerance` of each other, `MASK` takes an edge pixel's depth from its side of the depth mask's mask. The
    band (`near`, `far`) and the mask's reach are the depth mask's, which it is handed."""

    def __init__(self, settings: ViewDepthSettings, depth_mask: ViewDepthMask) -> None:
        self.settings: ViewDepthSettings = settings
        self._depth_mask: ViewDepthMask = depth_mask
        self._depth: Fbo = Fbo(interpolation=GL_NEAREST)     # an edge never reads as a distance in between
        self._shader: DepthUpscale = DepthUpscale()

    @property
    def depth(self) -> Texture:
        return self._depth

    @property
    def ports(self) -> dict[str, Port]:
        band: ViewDepthMaskSettings = self._depth_mask.settings
        return {'depth': Port(self._depth, TextureKind.DEPTH, depth_range=(band.near, band.far))}

    @property
    def made_with(self) -> tuple:
        """The settings its texture is made with, so its owner makes it again when one changes."""
        return (self.settings.depth_read, self.settings.tolerance)

    def allocate(self, width: int, height: int) -> None:
        self._depth.allocate(width, height, GL_R32F)
        self._shader.allocate()

    def deallocate(self) -> None:
        self._depth.deallocate()
        self._shader.deallocate()

    def clear(self) -> None:
        self._depth.clear()

    def read(self, depth: Texture | None, crop: Rect, read: DepthRead) -> None:
        """`depth` into the view across `crop` the way `read` says; none without a depth."""
        self._depth.clear()             # without a depth: none, not the last capture's
        if depth is None:
            return
        band: ViewDepthMaskSettings = self._depth_mask.settings
        self._depth.begin()
        self._shader.use(depth, crop, self.settings.tolerance, read, mask=self._depth_mask.mask,
                         near=band.near, far=band.far, reach=band.guided_radius)
        self._depth.end()
