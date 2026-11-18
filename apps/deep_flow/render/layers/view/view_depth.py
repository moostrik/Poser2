"""ViewDepth: how far. The camera's depth upscaled to the view once, blended within a surface, kept where it is the
player's and the mask is; then filled across the mask by push-pull, so the mask and the depth agree pixel for
pixel."""

# Third-party imports
from OpenGL.GL import GL_NEAREST, GL_R32F

# Local application imports
from modules.flow import DepthFill, DepthUpscale, Port, TextureKind
from modules.gl import Fbo, Texture
from modules.utils import Rect
from ....settings import ViewCropSettings

EDGE: float = 0.1       # m, the depth jump the upscale keeps as an edge: below a hand before the chest, as the motion's edge


class ViewDepth:
    """`upscale` brings a depth into the view across the crop: within a surface, the depth pixels within `EDGE` of
    each other blended; kept where it lies inside the box (the crop's near and far) and the mask is above 0; then the
    fill fills the mask where the upscaled depth has none (the band between the depth's coarse edge and the mask's,
    the camera's holes). It has no settings."""

    def __init__(self, box: ViewCropSettings) -> None:
        self._box: ViewCropSettings = box
        self._upscaled: Fbo = Fbo(interpolation=GL_NEAREST)      # an edge never reads as a distance in between
        self._filled: Fbo = Fbo(interpolation=GL_NEAREST)
        self._upscale_shader: DepthUpscale = DepthUpscale()
        self._fill: DepthFill = DepthFill()

    @property
    def depth(self) -> Texture:
        """The depth the view hands on, the upscaled depth filled."""
        return self._filled

    @property
    def ports(self) -> dict[str, Port]:
        band: tuple[float, float] = (self._box.near, self._box.far)
        return {'upscaled': Port(self._upscaled, TextureKind.DEPTH, depth_range=band),
                'filled': Port(self._filled, TextureKind.DEPTH, depth_range=band)}

    def allocate(self, width: int, height: int) -> None:
        self._upscaled.allocate(width, height, GL_R32F)
        self._filled.allocate(width, height, GL_R32F)
        self._upscale_shader.allocate()
        self._fill.allocate(width, height)

    def deallocate(self) -> None:
        self._upscaled.deallocate()
        self._filled.deallocate()
        self._upscale_shader.deallocate()
        self._fill.deallocate()

    def clear(self) -> None:
        self._upscaled.clear()
        self._filled.clear()

    def upscale(self, depth: Texture | None, crop: Rect, mask: Texture) -> None:
        """`depth` upscaled into the view across `crop`, where it is the player's and `mask` is, then filled across
        `mask`; none without a depth."""
        self.clear()                    # without a depth: none, not the last frame's
        if depth is None:
            return
        self._upscaled.begin()
        self._upscale_shader.use(depth, crop, EDGE, mask, self._box.near, self._box.far)
        self._upscaled.end()
        self._fill.use(self._upscaled, mask, self._filled)
