from OpenGL.GL import (Constant, GL_CLAMP_TO_EDGE, GL_COLOR_ATTACHMENT0, GL_FRAMEBUFFER, GL_LINEAR, glBindFramebuffer,
                       glDeleteFramebuffers, glFramebufferTexture, glGenFramebuffers, glViewport)

from .texture_2d_array import Texture2DArray


class LayeredFbo(Texture2DArray):
    """A 2D array texture with a framebuffer that draws into all its layers: a draw picks the layer per
    primitive (`gl_Layer`). `begin` binds it and sets the viewport to one layer's size."""

    def __init__(self, interpolation: int = GL_LINEAR, wrap: int = GL_CLAMP_TO_EDGE,
                 border_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)) -> None:
        super().__init__(interpolation, wrap, border_color)
        self.fbo_id: int = 0

    def allocate(self, width: int, height: int, depth: int, internal_format: Constant | int) -> None:
        super().allocate(width, height, depth, internal_format)
        if not self.allocated:
            return
        self.fbo_id = int(glGenFramebuffers(1))
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo_id)
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, self.tex_id, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def deallocate(self) -> None:
        if self.fbo_id:
            glDeleteFramebuffers(1, [self.fbo_id])
            self.fbo_id = 0
        super().deallocate()

    def begin(self) -> None:
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo_id)
        glViewport(0, 0, self.width, self.height)

    def end(self) -> None:
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
