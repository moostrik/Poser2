"""The fog: a volume drawn continuously in its box from the orbit camera, a ray per pixel, each sample
colored as the image or the colormap colors it and its amount its opacity; the box's edges over it.
Beside the stacked layers and the dots, it shows a thin sheet inside a dense volume."""

import numpy as np
from OpenGL.GL import GL_RGBA8

from modules.gl import Fbo, Style, Texture, Texture3D
from .orbit import Orbit, box_size
from .shaders import CubeAxes, CubeEdges, VolumeFogShader

STEPS: int = 96             # samples along a ray through the whole box
DENSITY: float = 6.0        # opacity per box width at an amount of 1


class VolumeFog:
    """Draws a volume's fog into its own FBO (RGBA8) at the size `resize` sets; render thread only."""

    def __init__(self) -> None:
        self._fbo: Fbo = Fbo()
        self._fog: VolumeFogShader = VolumeFogShader()
        self._edges: CubeEdges = CubeEdges()
        self._axes: CubeAxes = CubeAxes()

    @property
    def texture(self) -> Texture:
        return self._fbo

    def allocate(self) -> None:
        self._fog.allocate()
        self._edges.allocate()
        self._axes.allocate()

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._fog.deallocate()
        self._edges.deallocate()
        self._axes.deallocate()

    def resize(self, width: int, height: int) -> None:
        if self._fbo.allocated and (self._fbo.width, self._fbo.height) == (width, height):
            return
        self._fbo.deallocate()
        self._fbo.allocate(width, height, GL_RGBA8)

    def process(self, volume: Texture3D, orbit: Orbit, extent: float, scale: float, channel: int, colormap: bool) -> Texture:
        """Draw the fog of `volume`; returns its texture."""
        if not self._fbo.allocated or not volume.allocated:
            return self._fbo
        box: tuple[float, float, float] = box_size(volume.width, volume.height, extent)
        matrix: np.ndarray = orbit.view_projection(box, self._fbo.width / max(self._fbo.height, 1))
        self._fbo.clear(0.0, 0.0, 0.0, 0.0)
        self._fbo.begin()
        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)
        self._fog.reload()
        self._fog.use(volume, np.linalg.inv(matrix), box, scale, channel, colormap, STEPS, DENSITY)
        Style.set_blend_mode(Style.BlendMode.ALPHA)
        self._edges.reload()
        self._edges.use(matrix, box)
        self._axes.reload()
        self._axes.use(matrix, box)
        Style.pop_style()
        self._fbo.end()
        return self._fbo
