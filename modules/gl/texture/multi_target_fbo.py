"""MultiTargetFbo: a framebuffer that owns two or more color targets and draws into them at once; SwapMultiTargetFbo,
a ping-pong pair of them; SwapTarget, one target of such a pair, always its current framebuffer's."""

from collections.abc import Sequence

from OpenGL.GL import (Constant, GL_COLOR_ATTACHMENT0, GL_FRAMEBUFFER, GL_TEXTURE_2D, glBindFramebuffer,
                       glDeleteFramebuffers, glDrawBuffers, glFramebufferTexture2D, glGenFramebuffers, glViewport)

from .swap import Swap, TextureSwap
from .texture import Texture


class MultiTargetFbo:
    """A framebuffer over color targets it owns, one size: a draw writes its output `layout(location = i)` into
    target i. The caller makes the targets' `Texture`s (their filter and wrap) and gives them; `allocate` makes them at
    a format each, `deallocate` frees them, and each is handed out by index (`targets[i]`)."""

    def __init__(self, *targets: Texture) -> None:
        self.fbo_id: int = 0
        self.targets: tuple[Texture, ...] = targets

    @property
    def allocated(self) -> bool:
        return self.fbo_id != 0

    @property
    def width(self) -> int:
        return self.targets[0].width

    @property
    def height(self) -> int:
        return self.targets[0].height

    def allocate(self, width: int, height: int, formats: Sequence[Constant | int]) -> None:
        """Each target made at `width` × `height` in its format, in order, and the framebuffer over all of them."""
        self.deallocate()
        for texture, internal_format in zip(self.targets, formats, strict=True):
            texture.allocate(width, height, internal_format)
        attachments: list[int] = [int(GL_COLOR_ATTACHMENT0) + i for i in range(len(self.targets))]
        self.fbo_id = int(glGenFramebuffers(1))
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo_id)
        for attachment, texture in zip(attachments, self.targets):
            glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_2D, texture.tex_id, 0)
        glDrawBuffers(len(attachments), attachments)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def deallocate(self) -> None:
        """The framebuffer and its targets freed."""
        if self.fbo_id:
            glDeleteFramebuffers(1, [self.fbo_id])
            self.fbo_id = 0
        for texture in self.targets:
            texture.deallocate()

    def begin(self) -> None:
        """Draw into the targets, their whole size the viewport."""
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo_id)
        glViewport(0, 0, self.width, self.height)

    def end(self) -> None:
        """The window's framebuffer bound again."""
        glBindFramebuffer(GL_FRAMEBUFFER, 0)


class SwapMultiTargetFbo(Swap[MultiTargetFbo]):
    """A ping-pong pair of `MultiTargetFbo`s, swapped as one: `current` drawn into, `back` read. `target(i)` is
    target i of the pair, which stays its current framebuffer's across swaps."""

    def allocate(self, width: int, height: int, formats: Sequence[Constant | int]) -> None:
        """Both framebuffers made, their targets in `formats`."""
        for buffer in self._buffers:
            buffer.allocate(width, height, formats)

    def target(self, index: int) -> 'SwapTarget':
        return SwapTarget(self, index)


class SwapTarget(TextureSwap[Texture], Texture):
    """Target `index` of a `SwapMultiTargetFbo`: used as a texture it is the current framebuffer's, its `back` the
    other's, so a port or a pass keeps it as it keeps a `SwapFbo`. It swaps with its pair, never alone."""

    def __init__(self, pair: SwapMultiTargetFbo, index: int) -> None:
        self._pair: SwapMultiTargetFbo = pair
        self._index: int = index

    @property
    def current(self) -> Texture:
        return self._pair.current.targets[self._index]

    @property
    def back(self) -> Texture:
        return self._pair.back.targets[self._index]

    @property
    def allocated(self) -> bool:
        return self._pair.allocated

    def clear_all(self, r: float = 0.0, g: float = 0.0, b: float = 0.0, a: float = 0.0) -> None:
        """This target of both framebuffers cleared."""
        self.current.clear(r, g, b, a)
        self.back.clear(r, g, b, a)

    def set_wrap(self, wrap: int, border_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)) -> None:
        """This target's x and y wrap changed in both framebuffers (GL thread only)."""
        self.current.set_wrap(wrap, border_color)
        self.back.set_wrap(wrap, border_color)

    def swap(self) -> None:
        raise RuntimeError('a target swaps with its pair (SwapMultiTargetFbo.swap)')

    def allocate(self, width: int, height: int, internal_format: Constant | int) -> None:
        raise RuntimeError('a target is made with its pair (SwapMultiTargetFbo.allocate)')

    def deallocate(self) -> None:
        raise RuntimeError('a target is freed with its pair (SwapMultiTargetFbo.deallocate)')
