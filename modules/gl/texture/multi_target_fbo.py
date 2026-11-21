"""MultiTargetFbo: a framebuffer that draws into the textures it is given, two or more at once."""

from collections.abc import Sequence

from OpenGL.GL import (GL_COLOR_ATTACHMENT0, GL_FRAMEBUFFER, GL_TEXTURE_2D, glBindFramebuffer, glDeleteFramebuffers,
                       glDrawBuffers, glFramebufferTexture2D, glGenFramebuffers, glViewport)

from .texture import Texture


class MultiTargetFbo:
    """A framebuffer whose color targets are textures of the caller's, all of one size: a draw writes its output
    `layout(location = i)` into target i. It owns no texture; `Fbo` is the one texture that draws into itself."""

    def __init__(self) -> None:
        self.fbo_id: int = 0
        self._targets: tuple[Texture, ...] = ()

    @property
    def allocated(self) -> bool:
        return self.fbo_id != 0

    @property
    def width(self) -> int:
        return self._targets[0].width if self._targets else 0

    @property
    def height(self) -> int:
        return self._targets[0].height if self._targets else 0

    def allocate(self, targets: Sequence[Texture]) -> None:
        """The framebuffer over `targets`, allocated textures of one size, each its color attachment in order and
        all of them drawn into."""
        self.deallocate()
        self._targets = tuple(targets)
        attachments: list[int] = [int(GL_COLOR_ATTACHMENT0) + i for i in range(len(self._targets))]
        self.fbo_id = int(glGenFramebuffers(1))
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo_id)
        for attachment, texture in zip(attachments, self._targets):
            glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_2D, texture.tex_id, 0)
        glDrawBuffers(len(attachments), attachments)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def deallocate(self) -> None:
        """The framebuffer freed; the targets are the caller's and stay."""
        if self.fbo_id:
            glDeleteFramebuffers(1, [self.fbo_id])
            self.fbo_id = 0
        self._targets = ()

    def begin(self) -> None:
        """Draw into the targets, their whole size the viewport."""
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo_id)
        glViewport(0, 0, self.width, self.height)

    def end(self) -> None:
        """The window's framebuffer bound again."""
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
