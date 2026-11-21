from OpenGL.GL import (Constant, GL_CLAMP_TO_EDGE, GL_COLOR_ATTACHMENT0, GL_COLOR_BUFFER_BIT, GL_DEPTH_ATTACHMENT,
                       GL_DEPTH_BUFFER_BIT, GL_DEPTH_COMPONENT24, GL_FRAMEBUFFER, GL_LINEAR, GL_RENDERBUFFER, GL_TEXTURE_2D,
                       glBindFramebuffer, glBindRenderbuffer, glClear, glClearColor, glDeleteFramebuffers, glDeleteRenderbuffers,
                       glFramebufferRenderbuffer, glFramebufferTexture2D, glGenFramebuffers, glGenRenderbuffers,
                       glRenderbufferStorage, glViewport)
from .swap import TextureSwap
from .texture import Texture


class Fbo(Texture):
    """A framebuffer that owns its one color texture, the texture it is, and optionally a depth buffer."""

    def __init__(self,
                 interpolation: int = GL_LINEAR,
                 wrap: int = GL_CLAMP_TO_EDGE,
                 border_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
                 depth: bool = False) -> None:
        """`depth`: a depth buffer beside the color, for a draw with the depth test."""
        super().__init__(interpolation, wrap, border_color)
        self.fbo_id = 0
        self._depth: bool = depth
        self._depth_id: int = 0

    def allocate(self, width: int, height: int, internal_format: Constant | int) -> None:
        super().allocate(width, height, internal_format)
        if not self.allocated:
            return

        self.fbo_id = glGenFramebuffers(1)

        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo_id)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.tex_id, 0)
        if self._depth:
            self._depth_id = glGenRenderbuffers(1)
            glBindRenderbuffer(GL_RENDERBUFFER, self._depth_id)
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height)
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self._depth_id)
            glBindRenderbuffer(GL_RENDERBUFFER, 0)

        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(int(GL_COLOR_BUFFER_BIT) | (int(GL_DEPTH_BUFFER_BIT) if self._depth else 0))

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def deallocate(self) -> None:
        if self.fbo_id:
            glDeleteFramebuffers(1, [self.fbo_id])
            self.fbo_id = 0
        if self._depth_id:
            glDeleteRenderbuffers(1, [self._depth_id])
            self._depth_id = 0
        super().deallocate()

    def begin(self) -> None:
        """Draw into this FBO, its whole size the viewport."""
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo_id)
        glViewport(0, 0, self.width, self.height)

    def end(self) -> None:
        """End drawing into this FBO: the window's framebuffer is bound again."""
        glBindFramebuffer(GL_FRAMEBUFFER, 0)


class SwapFbo(TextureSwap[Fbo], Fbo):
    """Double-buffered FBO: drawn into, it is its current buffer."""

    def __init__(self,
                 interpolation: int = GL_LINEAR,
                 wrap: int = GL_CLAMP_TO_EDGE,
                 border_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)) -> None:
        TextureSwap.__init__(self, Fbo(interpolation, wrap, border_color), Fbo(interpolation, wrap, border_color))

    @property
    def fbo_id(self) -> int:
        return self.current.fbo_id

    def allocate(self, width: int, height: int, internal_format: Constant | int) -> None:
        """Both buffers made."""
        for buffer in self._buffers:
            buffer.allocate(width, height, internal_format)

    def begin(self) -> None:
        """Draw into the current buffer, its whole size the viewport."""
        self.current.begin()

    def end(self) -> None:
        """End rendering to current buffer."""
        self.current.end()
