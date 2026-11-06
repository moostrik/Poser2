"""The cube: a volume in 3D, its layers stacked in its box as the channel view draws them, from the
orbit camera, with the box's edges for its shape. The selected layer, the one the slice is at, is
drawn opaque and pulled out of the stack. One instanced draw, ordered far to near by the side of the
box the camera is on."""

from OpenGL.GL import GL_RGBA8

from modules.gl import Fbo, Style, Texture, Texture3D
from .orbit import Orbit, box_size
from .shaders import CubeEdges, CubeSlices

GAP: float = 3.0            # layer spacings on each side of the selected layer
OPACITY: float = 0.3        # the other layers'


class VolumeCube:
    """Draws a volume's cube into its own FBO (RGBA8) at the size `resize` sets; render thread only."""

    def __init__(self) -> None:
        self._fbo: Fbo = Fbo()
        self._slices: CubeSlices = CubeSlices()
        self._edges: CubeEdges = CubeEdges()

    @property
    def texture(self) -> Texture:
        return self._fbo

    def allocate(self) -> None:
        self._slices.allocate()
        self._edges.allocate()

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._slices.deallocate()
        self._edges.deallocate()

    def resize(self, width: int, height: int) -> None:
        if self._fbo.allocated and (self._fbo.width, self._fbo.height) == (width, height):
            return
        self._fbo.deallocate()
        self._fbo.allocate(width, height, GL_RGBA8)

    def process(self, volume: Texture3D, orbit: Orbit, position: float, extent: float,
                scale: float, channel: int, colormap: bool) -> Texture:
        """Draw the cube of `volume`, the layer at `position` (0 front, 1 back) pulled out; returns its texture."""
        if not self._fbo.allocated or not volume.allocated:
            return self._fbo
        box: tuple[float, float, float] = box_size(volume.width, volume.height, extent)
        matrix = orbit.view_projection(box, self._fbo.width / max(self._fbo.height, 1))
        selected: int = min(max(int(position * volume.depth), 0), volume.depth - 1)
        self._fbo.clear(0.0, 0.0, 0.0, 0.0)
        self._fbo.begin()
        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.ALPHA)
        self._slices.reload()
        self._slices.use(volume, matrix, box, orbit.front_on_near_side(), selected, GAP, scale, channel, colormap, OPACITY)
        self._edges.reload()
        self._edges.use(matrix, box)
        Style.pop_style()
        self._fbo.end()
        return self._fbo
