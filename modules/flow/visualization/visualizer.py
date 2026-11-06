"""The visualizer: draws a texture's channels into its own FBO, as an image, as a colormap centered on grey,
as arrows or as a histogram. A channel is 0 to 3 for x, y, z, w, or `ALL_CHANNELS`; a channel the
texture does not have reads as sampled: a one-channel texture repeats x in y and z (its grey
swizzle), and a missing w is 1."""

import numpy as np
from OpenGL.GL import GL_NEAREST, GL_R32UI, GL_RED_INTEGER, GL_RGBA8, GL_UNSIGNED_INT, glClearTexImage

from modules.gl import Fbo, Texture, Texture3D

from .. import FlowUtil
from .shaders import ChannelView, HistogramCount, HistogramCountVolume, HistogramDraw, VelocityArrowField
from .strip import ALL_CHANNELS

ARROW_WIDTH: float = 1.0 / 20.0     # the arrows' line width, a share of their spacing
HISTOGRAM_BINS: int = 256
HISTOGRAM_SAMPLES: int = 512        # samples per side at most: the texture's own grid up to this, a fixed cost above
HISTOGRAM_VOLUME_SAMPLES: int = 128 # per side across a volume, all its layers deep: about as many samples as a 2D cap


def histogram_grid(width: int, height: int) -> tuple[int, int]:
    """The grid of samples the histogram reads a texture of `width` × `height` on: its own, capped."""
    return (max(1, min(width, HISTOGRAM_SAMPLES)), max(1, min(height, HISTOGRAM_SAMPLES)))


def histogram_volume_grid(width: int, height: int, depth: int) -> tuple[int, int, int]:
    """The 3D grid of samples the histogram reads a volume on: capped across, every layer deep."""
    return (max(1, min(width, HISTOGRAM_VOLUME_SAMPLES)), max(1, min(height, HISTOGRAM_VOLUME_SAMPLES)), max(1, depth))


class Visualizer:
    """Draws one texture per call into its FBO, `texture`, at the size `resize` sets, the size the
    view is drawn at; render thread only."""

    def __init__(self) -> None:
        self._fbo: Fbo = Fbo()
        self._channel_view: ChannelView = ChannelView()
        self._arrows: VelocityArrowField = VelocityArrowField()
        self._histogram: Texture = Texture(GL_NEAREST)  # the bins, one row per channel; an integer texture must not filter
        self._histogram_count: HistogramCount = HistogramCount()
        self._histogram_count_volume: HistogramCountVolume = HistogramCountVolume()
        self._histogram_draw: HistogramDraw = HistogramDraw()
        self._histogram_total: int = 1                  # the samples counted last, a bin's count at full height
        self._zero: np.ndarray = np.zeros(1, dtype=np.uint32)

    @property
    def texture(self) -> Texture:
        """The texture as drawn last (RGBA8)."""
        return self._fbo

    @property
    def histogram_total(self) -> int:
        """The samples the last histogram counted, a bin's count at full height."""
        return self._histogram_total

    def resize(self, width: int, height: int) -> None:
        """Draw at `width` × `height` from now on; reallocates only when it changes."""
        if self._fbo.allocated and (self._fbo.width, self._fbo.height) == (width, height):
            return
        if self._fbo.allocated:
            self._fbo.deallocate()
        self._fbo.allocate(width, height, GL_RGBA8)

    def allocate(self) -> None:
        """The shaders and the histogram's bins; the FBO comes with the first `resize`."""
        self._histogram.allocate(HISTOGRAM_BINS, 4, GL_R32UI)
        self._channel_view.allocate()
        self._arrows.allocate()
        self._histogram_count.allocate()
        self._histogram_count_volume.allocate()
        self._histogram_draw.allocate()

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._histogram.deallocate()
        self._channel_view.deallocate()
        self._arrows.deallocate()
        self._histogram_count.deallocate()
        self._histogram_count_volume.deallocate()
        self._histogram_draw.deallocate()

    def process_image(self, texture: Texture, scale: float, channel: int = ALL_CHANNELS) -> None:
        """The texture as stored, times `scale`; one channel as grey."""
        self._process_channels(texture, scale, channel, False)

    def process_colormap(self, texture: Texture, scale: float, channel: int = ALL_CHANNELS) -> None:
        """The texture around mid grey, grey 0, times `scale`: x, y, z as red, green, blue; one
        channel in its own color, w as grey."""
        self._process_channels(texture, scale, channel, True)

    def process_arrows(self, texture: Texture, scale: float, spacing: float) -> None:
        """The arrow of x and y, with a ring for z when the texture has it, `spacing` apart (pixels):
        at speed 1 an arrow reaches one spacing, and at most two; its lines are spacing / 20 wide."""
        if not self._fbo.allocated or not texture.allocated:
            return
        xyz: bool = FlowUtil.get_num_channels(texture.internal_format) >= 3
        self._fbo.clear(0.0, 0.0, 0.0, 0.0)
        self._fbo.begin()
        self._arrows.reload()
        self._arrows.use(texture, scale, spacing, spacing * ARROW_WIDTH, xyz)
        self._fbo.end()

    def process_histogram(self, texture: Texture | Texture3D, scale: float, channel: int = ALL_CHANNELS) -> None:
        """How many of the texture's values fall where from -1 to 1 after `scale`, 0 in the middle,
        the height the log of the count: four lanes, or one channel's lane. Counted on the GPU over
        the texture's own grid of samples, capped at `HISTOGRAM_SAMPLES` per side; a volume through
        all its layers, capped at `HISTOGRAM_VOLUME_SAMPLES` across."""
        if not self._fbo.allocated or not texture.allocated or not self._histogram.allocated:
            return
        glClearTexImage(self._histogram.tex_id, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, self._zero)
        if isinstance(texture, Texture3D):
            grid3: tuple[int, int, int] = histogram_volume_grid(texture.width, texture.height, texture.depth)
            self._histogram_total = grid3[0] * grid3[1] * grid3[2]
            self._histogram_count_volume.reload()
            self._histogram_count_volume.use(texture, self._histogram, scale, grid3)
        else:
            grid: tuple[int, int] = histogram_grid(texture.width, texture.height)
            self._histogram_total = grid[0] * grid[1]
            self._histogram_count.reload()
            self._histogram_count.use(texture, self._histogram, scale, grid)
        self._fbo.begin()
        self._histogram_draw.reload()
        self._histogram_draw.use(self._histogram, channel, self._histogram_total)
        self._fbo.end()

    def _process_channels(self, texture: Texture, scale: float, channel: int, colormap: bool) -> None:
        if not self._fbo.allocated or not texture.allocated:
            return
        self._fbo.begin()
        self._channel_view.reload()
        self._channel_view.use(texture, scale, channel, colormap)
        self._fbo.end()
