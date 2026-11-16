"""ViewDepth: how far. The camera's depth read into the view once, blended within a surface, kept where it is the
player's and the mask is; then filled across the mask by push-pull, so the mask and the depth agree pixel for
pixel."""

# Third-party imports
from OpenGL.GL import GL_NEAREST, GL_R32F

# Local application imports
from modules.flow import DepthFill, DepthUpscale, Port, TextureKind
from modules.gl import Fbo, Texture
from modules.utils import Rect
from ....settings import ViewCropSettings

EDGE: float = 0.1       # m, the depth jump the read keeps as an edge: below a hand before the chest, as the motion's edge


class ViewDepth:
    """`read` brings a depth into the view across the crop: within a surface, the depth pixels within `EDGE` of each
    other blended; kept where it lies inside the box (the crop's near and far) and the mask is above 0; then the
    fill fills the mask where the read has none (the band between the depth's coarse edge and the mask's, the
    camera's holes). It has no settings."""

    def __init__(self, box: ViewCropSettings) -> None:
        self._box: ViewCropSettings = box
        self._read: Fbo = Fbo(interpolation=GL_NEAREST)      # an edge never reads as a distance in between
        self._depth: Fbo = Fbo(interpolation=GL_NEAREST)
        self._read_shader: DepthUpscale = DepthUpscale()
        self._fill: DepthFill = DepthFill()

    @property
    def depth(self) -> Texture:
        """The depth the view hands on, the read filled."""
        return self._depth

    @property
    def ports(self) -> dict[str, Port]:
        band: tuple[float, float] = (self._box.near, self._box.far)
        return {'read': Port(self._read, TextureKind.DEPTH, depth_range=band),
                'depth': Port(self._depth, TextureKind.DEPTH, depth_range=band)}

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
        self._read_shader.use(depth, crop, EDGE, mask, self._box.near, self._box.far)
        self._read.end()
        self._fill.use(self._read, mask, self._depth)
