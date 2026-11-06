"""The strip: a texture's channels over time, a rolling plot of the frame-wide mean amount (absolute
value) of x, y, z and w, or of one of them, one column per tick, on a log axis of three decades. It stays on the GPU, so the render thread never waits on a read."""

from OpenGL.GL import GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR, GL_RGBA16F, GL_RGBA8, GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER, glBindTexture, glGenerateMipmap, glTexParameteri

from modules.gl import Fbo, SwapFbo, Texture
from .. import FlowUtil
from .shaders import StripRoll, StripSummary

LANES: tuple[str, ...] = ('x', 'y', 'z', 'w')      # top to bottom
ALL_CHANNELS: int = 4           # the channel that stands for all four
STRIP_WIDTH: int = 540          # columns: 9 s at 60 ticks per second; the height is the size it is drawn at
SUMMARY_LIMIT: int = 512        # the summary reads the texture at its own size up to this per side


def summary_grid(width: int, height: int) -> tuple[int, int]:
    """The grid the strip's summary reads a texture of `width` × `height` on: its own, capped."""
    return (max(1, min(width, SUMMARY_LIMIT)), max(1, min(height, SUMMARY_LIMIT)))


class Strip:
    """Rolling plot of a texture's channels; advance it once per tick with the texture to plot. Its
    columns are its history, its height the one `resize` sets."""

    def __init__(self) -> None:
        self._summary_fbo: Fbo = Fbo()
        self._strip_fbo: SwapFbo = SwapFbo()
        self._summary_shader: StripSummary = StripSummary()
        self._roll_shader: StripRoll = StripRoll()

    @property
    def texture(self) -> Texture:
        """The plot (RGBA8)."""
        return self._strip_fbo

    def allocate(self) -> None:
        """The shaders; the plot comes with the first `resize`, the summary with the first texture."""
        self._summary_shader.allocate()
        self._roll_shader.allocate()

    def deallocate(self) -> None:
        self._summary_fbo.deallocate()
        self._strip_fbo.deallocate()
        self._summary_shader.deallocate()
        self._roll_shader.deallocate()

    def resize(self, height: int) -> None:
        """Plot at `height` from now on; a change starts the plot over."""
        if self._strip_fbo.allocated and self._strip_fbo.height == height:
            return
        if self._strip_fbo.allocated:
            self._strip_fbo.deallocate()
        self._strip_fbo.allocate(STRIP_WIDTH, height, GL_RGBA8)
        FlowUtil.zero(self._strip_fbo)

    def reset(self) -> None:
        if self._strip_fbo.allocated:
            FlowUtil.zero(self._strip_fbo)

    def _follow_summary(self, texture: Texture) -> None:
        """The summary at the texture's own grid, capped; mipmapped, so its top level is the mean."""
        grid: tuple[int, int] = summary_grid(texture.width, texture.height)
        if self._summary_fbo.allocated and (self._summary_fbo.width, self._summary_fbo.height) == grid:
            return
        if self._summary_fbo.allocated:
            self._summary_fbo.deallocate()
        self._summary_fbo.allocate(*grid, GL_RGBA16F)
        glBindTexture(GL_TEXTURE_2D, self._summary_fbo.tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glGenerateMipmap(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, 0)

    def update(self, texture: Texture, scale: float, channel: int = ALL_CHANNELS) -> None:
        """Add one column: the mean amount of each of `texture`'s channels, times `scale`, in four lanes,
        or of `channel` (0 to 3) alone at full height."""
        if not self._strip_fbo.allocated or not texture.allocated:
            return
        self._follow_summary(texture)
        self._summary_fbo.begin()
        self._summary_shader.use(texture)
        self._summary_fbo.end()
        glBindTexture(GL_TEXTURE_2D, self._summary_fbo.tex_id)
        glGenerateMipmap(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, 0)

        self._strip_fbo.swap()
        self._strip_fbo.begin()
        self._roll_shader.use(self._strip_fbo.back, self._summary_fbo, scale, channel)
        self._strip_fbo.end()
