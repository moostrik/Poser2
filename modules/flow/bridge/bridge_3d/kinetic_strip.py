"""Kinetic Strip.

The kinetic bridge over time: a rolling plot of the frame-wide mean of speed, onset, release and
emission, one column per optical flow frame. It stays on the GPU, so the render thread never waits on a read.
"""
from OpenGL.GL import GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR, GL_RGBA16F, GL_RGBA8, GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER, glBindTexture, glGenerateMipmap, glTexParameteri

from modules.gl import Fbo, SwapFbo, Texture
from ... import FlowUtil
from .shaders import KineticStripShader, KineticSummary

LANES: tuple[str, ...] = ('speed', 'onset', 'release', 'emission')      # top to bottom
STRIP_WIDTH: int = 540          # columns: 9 s at 60 flow frames per second
STRIP_HEIGHT: int = 960
SUMMARY_WIDTH: int = 128        # enough samples for a mean of fields with thin edges
SUMMARY_HEIGHT: int = 228


class KineticStrip:
    """Rolling plot of the kinetic bridge's fields; advance it on each new optical flow frame."""

    def __init__(self) -> None:
        self._summary_fbo: Fbo = Fbo()
        self._strip_fbo: SwapFbo = SwapFbo()
        self._summary_shader: KineticSummary = KineticSummary()
        self._strip_shader: KineticStripShader = KineticStripShader()

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
        self._strip_shader.allocate()

    def deallocate(self) -> None:
        self._summary_fbo.deallocate()
        self._strip_fbo.deallocate()
        self._summary_shader.deallocate()
        self._strip_shader.deallocate()

    def reset(self) -> None:
        FlowUtil.zero(self._strip_fbo)

    def update(self, fast: Texture, acceleration: Texture, emission: Texture, scale: float) -> None:
        """Add one column from the fields of this optical flow frame."""
        self._summary_fbo.begin()
        self._summary_shader.use(fast, acceleration, emission)
        self._summary_fbo.end()
        glBindTexture(GL_TEXTURE_2D, self._summary_fbo.tex_id)
        glGenerateMipmap(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, 0)

        self._strip_fbo.swap()
        self._strip_fbo.begin()
        self._strip_shader.use(self._strip_fbo.back_texture, self._summary_fbo, scale)
        self._strip_fbo.end()
