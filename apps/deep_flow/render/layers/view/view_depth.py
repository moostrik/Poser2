"""ViewDepth: how far. The camera's depth read into the view once, blended within a surface, at the outline from the
mask's side; and the depth mask, that depth cut to the player's band, on the mask's outline."""

# Third-party imports
from OpenGL.GL import GL_NEAREST, GL_R16F, GL_R32F

# Local application imports
from modules.flow import DepthUpscale, Port, TextureKind
from modules.gl import Fbo, Texture
from modules.utils import Rect
from ....settings import ViewDepthMaskSettings, ViewDepthSettings
from .shaders import DepthMask
from .view_depth_mask import ViewDepthMask


class ViewDepth:
    """`read` brings a depth into the view, cut to the crop: within a surface, the depth pixels within `tolerance`
    of each other blended; at the outline, a pixel's depth from the side of the depth mask's mask it lies on, within
    the mask's reach (`guided_radius`). Then the depth mask: where that depth lies in the band. The band and the mask
    are the depth mask's, which it is handed."""

    def __init__(self, settings: ViewDepthSettings, depth_mask: ViewDepthMask) -> None:
        self.settings: ViewDepthSettings = settings
        self._who: ViewDepthMask = depth_mask
        self._depth: Fbo = Fbo(interpolation=GL_NEAREST)     # an edge never reads as a distance in between
        self._depth_mask: Fbo = Fbo()
        self._read_shader: DepthUpscale = DepthUpscale()
        self._cut_shader: DepthMask = DepthMask()

    @property
    def depth(self) -> Texture:
        return self._depth

    @property
    def depth_mask(self) -> Texture:
        return self._depth_mask

    @property
    def ports(self) -> dict[str, Port]:
        band: ViewDepthMaskSettings = self._who.settings
        return {'depth': Port(self._depth, TextureKind.DEPTH, depth_range=(band.near, band.far)),
                'depth_mask': Port(self._depth_mask, TextureKind.SCALAR)}

    @property
    def made_with(self) -> tuple:
        """The settings its textures are made with, so its owner makes them again when one changes."""
        return (self.settings.tolerance,)

    def allocate(self, width: int, height: int) -> None:
        self._depth.allocate(width, height, GL_R32F)
        self._depth_mask.allocate(width, height, GL_R16F)
        self._read_shader.allocate()
        self._cut_shader.allocate()

    def deallocate(self) -> None:
        self._depth.deallocate()
        self._depth_mask.deallocate()
        self._read_shader.deallocate()
        self._cut_shader.deallocate()

    def clear(self) -> None:
        self._depth.clear()
        self._depth_mask.clear()

    def read(self, depth: Texture | None, crop: Rect) -> None:
        """`depth` into the view across `crop`, from the mask's side at the outline, then its cut to the band; none
        without a depth."""
        self.clear()                    # without a depth: none, not the last capture's
        if depth is None:
            return
        band: ViewDepthMaskSettings = self._who.settings
        self._depth.begin()
        self._read_shader.use(depth, crop, self.settings.tolerance, self._who.mask, band.near, band.far, band.guided_radius)
        self._depth.end()
        self._depth_mask.begin()
        self._cut_shader.use(self._depth, band.near, band.far)
        self._depth_mask.end()
