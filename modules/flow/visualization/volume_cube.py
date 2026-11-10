"""The cube: a volume drawn continuously in its box from the orbit camera, a ray per pixel, each sample
colored by its kind (`CubeColor`) and its amount its opacity; with `outlines` the box's edges, its axes, the slice's frame and the
light's place (`light_marks`), in the scene. With a
slice, the selected layer is read where a ray crosses it and framed, opacity moved from the rest to it by `slice_focus`."""

import numpy as np
from OpenGL.GL import GL_RGBA8

from modules.gl import Fbo, Style, Texture, Texture3D
from ..port import LightPlace
from .light_marks import light_marks
from .orbit import Orbit, box_size
from .shaders import CubeAxes, CubeColor, CubeEdges, VolumeCubeShader

STEPS: int = 96             # samples along a ray through the whole box
FRAME_COLOR: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)     # the slice's frame
LIGHT_COLOR: tuple[float, float, float, float] = (1.0, 0.757, 0.027, 1.0)  # the light's marks: amber, the light settings' color


class VolumeCube:
    """Draws a volume's cube into its own FBO (RGBA8) at the size `resize` sets; render thread only."""

    def __init__(self) -> None:
        self._fbo: Fbo = Fbo()
        self._cube: VolumeCubeShader = VolumeCubeShader()
        self._edges: CubeEdges = CubeEdges()
        self._axes: CubeAxes = CubeAxes()

    @property
    def texture(self) -> Texture:
        return self._fbo

    def allocate(self) -> None:
        self._cube.allocate()
        self._edges.allocate()
        self._axes.allocate()

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._cube.deallocate()
        self._edges.deallocate()
        self._axes.deallocate()

    def resize(self, width: int, height: int) -> None:
        if self._fbo.allocated and (self._fbo.width, self._fbo.height) == (width, height):
            return
        self._fbo.deallocate()
        self._fbo.allocate(width, height, GL_RGBA8)

    def process(self, volume: Texture3D, orbit: Orbit, extent: float, scale: float, channel: int, color: CubeColor,
                opacity: float, depth_fade: float, slice_z: float | None, outlines: bool = True,
                slice_focus: float = 0.5, light: LightPlace | None = None) -> Texture:
        """Draw the cube of `volume`, the layer at `slice_z` (0 front to 1 back) in focus unless None, and
        with `outlines` where `light` is; returns its texture."""
        if not self._fbo.allocated or not volume.allocated:
            return self._fbo
        box: tuple[float, float, float] = box_size(volume.width, volume.height, extent)
        matrix: np.ndarray = orbit.view_projection(box, self._fbo.width / max(self._fbo.height, 1))
        self._fbo.clear(0.0, 0.0, 0.0, 0.0)
        self._fbo.begin()
        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)
        self._cube.reload()
        self._cube.use(volume, np.linalg.inv(matrix), box, scale, channel, color, STEPS, opacity, depth_fade, slice_z, slice_focus)
        Style.set_blend_mode(Style.BlendMode.ALPHA)
        if outlines:
            self._edges.reload()
            self._edges.use(matrix, box)
            if slice_z is not None:         # the layer's frame: the box's edges flattened to its depth
                at_layer: np.ndarray = np.eye(4)
                at_layer[2, 3] = box[2] * (0.5 - slice_z)
                self._edges.use(matrix @ at_layer, (box[0], box[1], 0.0), FRAME_COLOR)
            if light is not None:
                for mark in light_marks(light, box):
                    self._edges.use(matrix @ mark, (1.0, 1.0, 1.0), LIGHT_COLOR)
            self._axes.reload()
            self._axes.use(matrix, box)
        Style.pop_style()
        self._fbo.end()
        return self._fbo
