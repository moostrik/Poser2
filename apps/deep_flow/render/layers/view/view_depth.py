"""ViewDepth: how far. The camera's depth read into the view once, blended within a surface, kept where it is the
player's and the mask is; then filled across the mask by push-pull, so the mask and the depth agree pixel for
pixel."""

# Third-party imports
from OpenGL.GL import GL_NEAREST, GL_R32F

# Local application imports
from modules.flow import DepthFill, DepthUpscale, Port, TextureKind
from modules.gl import Fbo, Texture
from modules.utils import Rect
from ....settings import ViewCropSettings, ViewDepthSettings


class ViewDepth:
    """`read` brings a depth into the view across the crop: within a surface, the depth pixels within `tolerance` of
    each other blended; kept where it lies inside the box (the crop's near and far) and the mask is above 0. `fill`
    fills the mask where the read has none (the band between the depth's coarse edge and the mask's, the camera's
    holes), with `fill_enabled`; off, the depth is the read."""

    def __init__(self, settings: ViewDepthSettings, box: ViewCropSettings) -> None:
        self.settings: ViewDepthSettings = settings
        self._box: ViewCropSettings = box
        self._read: Fbo = Fbo(interpolation=GL_NEAREST)      # an edge never reads as a distance in between
        self._depth: Fbo = Fbo(interpolation=GL_NEAREST)
        self._read_shader: DepthUpscale = DepthUpscale()
        self._fill: DepthFill = DepthFill()

    @property
    def depth(self) -> Texture:
        """The depth the view hands on: the filled one, or the read with the fill off."""
        return self._depth if self.settings.fill_enabled else self._read

    @property
    def ports(self) -> dict[str, Port]:
        band: tuple[float, float] = (self._box.near, self._box.far)
        return {'read': Port(self._read, TextureKind.DEPTH, depth_range=band),
                'depth': Port(self.depth, TextureKind.DEPTH, depth_range=band)}

    @property
    def made_with(self) -> tuple:
        """The settings its textures are made with, so its owner makes them again when one changes."""
        return (self.settings.tolerance, self.settings.fill_enabled)

    def allocate(self, width: int, height: int) -> None:
        self._read.allocate(width, height, GL_R32F)
        self._depth.allocate(width, height, GL_R32F)
        self._read_shader.allocate()
        self._fill.allocate(width, height)

    def deallocate(self) -> None:
        self._read.deallocate()
        self._depth.deallocate()
        self._read_shader.deallocate()
        self._fill.deallocate()

    def clear(self) -> None:
        self._read.clear()
        self._depth.clear()

    def read(self, depth: Texture | None, crop: Rect, mask: Texture) -> None:
        """`depth` into the view across `crop`, where it is the player's and `mask` is, then filled across `mask`;
        none without a depth."""
        self.clear()                    # without a depth: none, not the last capture's
        if depth is None:
            return
        self._read.begin()
        self._read_shader.use(depth, crop, self.settings.tolerance, mask, self._box.near, self._box.far)
        self._read.end()
        if self.settings.fill_enabled:
            self._fill.use(self._read, mask, self._depth)
