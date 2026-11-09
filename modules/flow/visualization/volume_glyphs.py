"""The 3D glyphs: a volume in its box as marks on a grid, from the orbit camera, with the depth test:
lit arrows for a velocity, dots for a scalar (sized and colored by the value), and the box's edges.
The grid is the glyph spacing across the panel, as deep as the volume's layers."""

import math

from OpenGL.GL import GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST, GL_LESS, GL_RGBA8, glClear, glDepthFunc, glDisable, glEnable

from modules.gl import Fbo, Style, Texture, Texture3D
from .. import FlowUtil
from .orbit import FOV, Orbit, box_size
from .shaders import CubeAxes, CubeEdges, GlyphArrows, GlyphDots

GRID_LIMIT: int = 96        # grid points across at most


def glyph_grid(width_px: int, spacing: float, box: tuple[float, float, float], layers: int) -> tuple[int, int, int]:
    """The grid points across (the panel's width over the spacing), down (in the box's proportions) and deep (the layers)."""
    across: int = min(max(round(width_px / max(spacing, 1.0)), 2), GRID_LIMIT)
    return (across, max(round(across * box[1] / box[0]), 2), max(layers, 1))


class VolumeGlyphs:
    """Draws a volume's 3D glyphs into its own FBO (RGBA8 with depth) at the size `resize` sets; render thread only."""

    def __init__(self) -> None:
        self._fbo: Fbo = Fbo(depth=True)
        self._arrows: GlyphArrows = GlyphArrows()
        self._dots: GlyphDots = GlyphDots()
        self._edges: CubeEdges = CubeEdges()
        self._axes: CubeAxes = CubeAxes()

    @property
    def texture(self) -> Texture:
        return self._fbo

    def allocate(self) -> None:
        self._arrows.allocate()
        self._dots.allocate()
        self._edges.allocate()
        self._axes.allocate()

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._arrows.deallocate()
        self._dots.deallocate()
        self._edges.deallocate()
        self._axes.deallocate()

    def resize(self, width: int, height: int) -> None:
        if self._fbo.allocated and (self._fbo.width, self._fbo.height) == (width, height):
            return
        self._fbo.deallocate()
        self._fbo.allocate(width, height, GL_RGBA8)

    def process(self, volume: Texture3D, orbit: Orbit, extent: float, spacing: float, scale: float,
                dots: bool, channel: int, push: bool = False, outlines: bool = True) -> Texture:
        """Draw arrows of the volume's velocity, with `push` an acceleration's cones, or with `dots`, or on a
        one-channel volume, which has no direction, its scalar's dots; returns the texture."""
        dots = dots or FlowUtil.get_num_channels(volume.internal_format) == 1
        if not self._fbo.allocated or not volume.allocated:
            return self._fbo
        box: tuple[float, float, float] = box_size(volume.width, volume.height, extent)
        matrix = orbit.view_projection(box, self._fbo.width / max(self._fbo.height, 1))
        grid: tuple[int, int, int] = glyph_grid(self._fbo.width, spacing, box, volume.depth)
        self._fbo.clear(0.0, 0.0, 0.0, 0.0)
        self._fbo.begin()
        glClear(GL_DEPTH_BUFFER_BIT)
        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        pixels_per_unit: float = self._fbo.height * 0.5 / math.tan(FOV / 2.0)
        if dots:
            self._dots.reload()
            self._dots.use(volume, matrix, box, grid, scale, channel, pixels_per_unit)
        else:
            self._arrows.reload()
            self._arrows.use(volume, matrix, box, grid, scale, pixels_per_unit, push)
        if outlines:                        # the box's edges and its axes, in the scene's depth
            self._edges.reload()
            self._edges.use(matrix, box)
            self._axes.reload()
            self._axes.use(matrix, box)
        glDisable(GL_DEPTH_TEST)
        Style.pop_style()
        self._fbo.end()
        return self._fbo
