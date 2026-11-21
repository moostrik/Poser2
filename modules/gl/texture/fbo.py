"""Fbo: a framebuffer that owns its color targets, one or more, and is its target 0; SwapFbo, a ping-pong pair of
them; SwapTarget, one target of such a pair, always its current framebuffer's."""

from collections.abc import Sequence

from OpenGL.GL import (Constant, GL_CLAMP_TO_EDGE, GL_COLOR_ATTACHMENT0, GL_COLOR_BUFFER_BIT, GL_DEPTH_ATTACHMENT,
                       GL_DEPTH_BUFFER_BIT, GL_DEPTH_COMPONENT24, GL_FRAMEBUFFER, GL_LINEAR, GL_RENDERBUFFER, GL_TEXTURE_2D,
                       glBindFramebuffer, glBindRenderbuffer, glClear, glClearColor, glDeleteFramebuffers, glDeleteRenderbuffers,
                       glDrawBuffers, glFramebufferRenderbuffer, glFramebufferTexture2D, glGenFramebuffers, glGenRenderbuffers,
                       glRenderbufferStorage, glViewport)

from .swap import TextureSwap
from .texture import Texture


class Fbo(Texture):
    """A framebuffer over color targets it owns, one size, and optionally a depth buffer. It is its target 0, the
    texture a pass writes; a draw writes its output `layout(location = i)` into target i, `targets[i]`."""

    def __init__(self,
                 interpolation: int = GL_LINEAR,
                 wrap: int = GL_CLAMP_TO_EDGE,
                 border_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
                 depth: bool = False,
                 extra_targets: Sequence[int] = ()) -> None:
        """`depth`: a depth buffer beside the color, for a draw with the depth test; `extra_targets`: the interpolation
        of each target after target 0 (GL_LINEAR or GL_NEAREST), at the same wrap."""
        super().__init__(interpolation, wrap, border_color)
        self.fbo_id = 0
        self._depth: bool = depth
        self._depth_id: int = 0
        self.targets: tuple[Texture, ...] = (self, *(Texture(i, wrap, border_color) for i in extra_targets))

    def allocate(self, width: int, height: int, internal_format: Constant | int | Sequence[Constant | int]) -> None:
        """The targets made at `width` × `height`: one format for the one target, a format each for several."""
        formats: Sequence[Constant | int] = internal_format if isinstance(internal_format, Sequence) else (internal_format,)
        if len(formats) != len(self.targets):
            raise ValueError(f'{len(formats)} formats for {len(self.targets)} targets')
        super().allocate(width, height, formats[0])
        if not self.allocated:
            return
        for texture, target_format in zip(self.targets[1:], formats[1:]):
            texture.allocate(width, height, target_format)

        self.fbo_id = glGenFramebuffers(1)

        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo_id)
        for i, texture in enumerate(self.targets):
            glFramebufferTexture2D(GL_FRAMEBUFFER, int(GL_COLOR_ATTACHMENT0) + i, GL_TEXTURE_2D, texture.tex_id, 0)
        if len(self.targets) > 1:
            glDrawBuffers(len(self.targets), [int(GL_COLOR_ATTACHMENT0) + i for i in range(len(self.targets))])
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
        for texture in self.targets[1:]:
            texture.deallocate()
        super().deallocate()

    def begin(self) -> None:
        """Draw into the targets, their whole size the viewport."""
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo_id)
        glViewport(0, 0, self.width, self.height)

    def end(self) -> None:
        """End drawing into this FBO: the window's framebuffer is bound again."""
        glBindFramebuffer(GL_FRAMEBUFFER, 0)


class SwapFbo(TextureSwap[Fbo], Fbo):
    """A ping-pong pair of `Fbo`s, swapped as one: drawn into, it is its current buffer. `target(i)` is target i of
    the pair, which stays its current framebuffer's across swaps."""

    def __init__(self,
                 interpolation: int = GL_LINEAR,
                 wrap: int = GL_CLAMP_TO_EDGE,
                 border_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
                 extra_targets: Sequence[int] = ()) -> None:
        """`extra_targets`: the interpolation of each target after target 0, as an `Fbo`'s."""
        TextureSwap.__init__(self, Fbo(interpolation, wrap, border_color, extra_targets=extra_targets),
                             Fbo(interpolation, wrap, border_color, extra_targets=extra_targets))

    @property
    def fbo_id(self) -> int:
        return self.current.fbo_id

    def allocate(self, width: int, height: int, internal_format: Constant | int | Sequence[Constant | int]) -> None:
        """Both buffers made."""
        for buffer in self._buffers:
            buffer.allocate(width, height, internal_format)

    def target(self, index: int) -> 'SwapTarget':
        return SwapTarget(self, index)

    def begin(self) -> None:
        """Draw into the current buffer, its whole size the viewport."""
        self.current.begin()

    def end(self) -> None:
        """End rendering to current buffer."""
        self.current.end()


class SwapTarget(TextureSwap[Texture], Texture):
    """Target `index` of a `SwapFbo`: used as a texture it is the current framebuffer's, its `back` the other's, so a
    port or a pass keeps it as it keeps the `SwapFbo`. It swaps with its pair, never alone."""

    def __init__(self, pair: SwapFbo, index: int) -> None:
        self._pair: SwapFbo = pair
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
        raise RuntimeError('a target swaps with its pair (SwapFbo.swap)')

    def allocate(self, width: int, height: int, internal_format: Constant | int) -> None:
        raise RuntimeError('a target is made with its pair (SwapFbo.allocate)')

    def deallocate(self) -> None:
        raise RuntimeError('a target is freed with its pair (SwapFbo.deallocate)')
