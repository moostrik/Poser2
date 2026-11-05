"""The strip: a texture's four channels over time, a rolling plot of the frame-wide mean amount of
x, y, z and w, one column per tick. It stays on the GPU, so the render thread never waits on a read."""

from OpenGL.GL import GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR, GL_RGBA16F, GL_RGBA8, GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER, glBindTexture, glGenerateMipmap, glTexParameteri

from modules.gl import Fbo, SwapFbo, Texture
from .. import FlowUtil
from .shaders import StripRoll, StripSummary

LANES: tuple[str, ...] = ('x', 'y', 'z', 'w')      # top to bottom
STRIP_WIDTH: int = 540          # columns: 9 s at 60 ticks per second
STRIP_HEIGHT: int = 960
SUMMARY_WIDTH: int = 128        # enough samples for a mean of fields with thin edges
SUMMARY_HEIGHT: int = 228


class Strip:
    """Rolling plot of a texture's channels; advance it once per tick with the texture to plot."""

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
        self._summary_fbo.allocate(SUMMARY_WIDTH, SUMMARY_HEIGHT, GL_RGBA16F)
        glBindTexture(GL_TEXTURE_2D, self._summary_fbo.tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glGenerateMipmap(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, 0)
        self._strip_fbo.allocate(STRIP_WIDTH, STRIP_HEIGHT, GL_RGBA8)
        FlowUtil.zero(self._strip_fbo)
        self._summary_shader.allocate()
        self._roll_shader.allocate()

    def deallocate(self) -> None:
        self._summary_fbo.deallocate()
        self._strip_fbo.deallocate()
        self._summary_shader.deallocate()
        self._roll_shader.deallocate()

    def reset(self) -> None:
        FlowUtil.zero(self._strip_fbo)

    def update(self, texture: Texture, scale: float) -> None:
        """Add one column: the mean amount of each of `texture`'s channels, times `scale`."""
        self._summary_fbo.begin()
        self._summary_shader.use(texture)
        self._summary_fbo.end()
        glBindTexture(GL_TEXTURE_2D, self._summary_fbo.tex_id)
        glGenerateMipmap(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, 0)

        self._strip_fbo.swap()
        self._strip_fbo.begin()
        self._roll_shader.use(self._strip_fbo.back, self._summary_fbo, scale)
        self._strip_fbo.end()
