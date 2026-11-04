from OpenGL.GL import Constant, GL_CLAMP_TO_EDGE, GL_COLOR_ATTACHMENT0, GL_COLOR_BUFFER_BIT, GL_FRAMEBUFFER, GL_LINEAR, GL_TEXTURE_2D, glBindFramebuffer, glClear, glClearColor, glFramebufferTexture2D, glGenFramebuffers, glViewport
from .texture import Texture

_current_viewport: tuple[int, int, int, int] = (0, 0, 0, 0)

class Fbo(Texture):
    def __init__(self,
                 interpolation: int = GL_LINEAR,
                 wrap: int = GL_CLAMP_TO_EDGE,
                 border_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)) -> None:
        super().__init__(interpolation, wrap, border_color)
        self.fbo_id = 0


    def allocate(self, width: int, height: int, internal_format) -> None:
        super().allocate(width, height, internal_format)
        if not self.allocated: return

        self.fbo_id = glGenFramebuffers(1)

        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo_id)
        self.bind()
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.tex_id, 0)

        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        self.unbind()

    def begin(self)  -> None:
        """Begin rendering to FBO. Uses top-left origin (see COORDINATE_SYSTEM.md)."""
        global _current_viewport

        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo_id)
        new_vp = (0, 0, self.width, self.height)
        if _current_viewport != new_vp:
            glViewport(*new_vp)
            _current_viewport = new_vp

        glViewport(*new_vp)

    def end(self)  -> None:
        """End rendering to FBO and restore previous state."""
        # pass
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

class StaleTextureError(RuntimeError):
    """A SwapFbo buffer was used after its owner swapped: it holds another frame than it did."""


class SwapTexture(Texture):
    """One buffer of a SwapFbo as handed out by `texture` or `back_texture`: valid until the owner's
    next swap. It reads as the buffer it was taken from; after a swap, a GL-facing use (its tex_id,
    bind) raises `StaleTextureError`, so a buffer kept across a swap fails where it is used instead
    of showing the image of another frame. Hold the owner and take its texture where it is used."""

    def __init__(self, owner: 'SwapFbo', buffer: Fbo, swaps: int) -> None:
        # no Texture.__init__: every field reads through to the buffer
        self._owner: SwapFbo = owner
        self._buffer: Fbo = buffer
        self._swaps: int = swaps

    def _checked(self) -> Fbo:
        if self._owner._swaps != self._swaps:
            raise StaleTextureError("a SwapFbo buffer was used after its owner swapped: hold the owner and read its "
                                    ".texture where it is used, never keep the texture")
        return self._buffer

    @property
    def tex_id(self) -> int:
        return self._checked().tex_id

    @property
    def fbo_id(self) -> int:
        return self._checked().fbo_id

    @property
    def width(self) -> int:
        return self._buffer.width

    @property
    def height(self) -> int:
        return self._buffer.height

    @property
    def internal_format(self) -> Constant:
        return self._buffer.internal_format

    @property
    def format(self) -> Constant:
        return self._buffer.format

    @property
    def data_type(self) -> Constant:
        return self._buffer.data_type

    @property
    def allocated(self) -> bool:
        return self._buffer.allocated

    def bind(self) -> None:
        self._checked().bind()

    def unbind(self) -> None:
        self._buffer.unbind()

    def __getattr__(self, name: str):
        # anything else a Texture has (read_to_numpy, set_wrap, ...), from the checked buffer
        return getattr(self._checked(), name)


class SwapFbo(Fbo):
    """Double-buffered FBO that swaps between two buffers.

    Inherits from Fbo and delegates all properties to the current write buffer.
    Use .texture for current buffer and .back_texture for previous buffer.
    """

    def __init__(self,
                 interpolation: int = GL_LINEAR,
                 wrap: int = GL_CLAMP_TO_EDGE,
                 border_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)) -> None:
        # Don't call super().__init__() - we manage internal FBOs
        self._fbos: list[Fbo] = [Fbo(interpolation, wrap, border_color), Fbo(interpolation, wrap, border_color)]
        self._swap_state: int = 0
        self._swaps: int = 0        # counted, so a buffer handed out before a swap knows it is stale

    # Override all Fbo/Texture properties to delegate to current buffer
    @property
    def fbo_id(self) -> int:
        return self._fbos[self._swap_state].fbo_id

    @property
    def tex_id(self) -> int:
        return self._fbos[self._swap_state].tex_id

    @property
    def width(self) -> int:
        return self._fbos[self._swap_state].width

    @property
    def height(self) -> int:
        return self._fbos[self._swap_state].height

    @property
    def internal_format(self) -> Constant:
        return self._fbos[self._swap_state].internal_format

    @property
    def format(self) -> Constant:
        return self._fbos[self._swap_state].format

    @property
    def data_type(self) -> Constant:
        return self._fbos[self._swap_state].data_type

    @property
    def allocated(self) -> bool:
        return self._fbos[0].allocated and self._fbos[1].allocated

    @property
    def texture(self) -> Texture:
        """Current write buffer, valid until the next swap (`SwapTexture`). The SwapFbo itself, used as
        a Texture, is always the current buffer."""
        return SwapTexture(self, self._fbos[self._swap_state], self._swaps)

    @property
    def back_texture(self) -> Texture:
        """Previous buffer for reading in ping-pong rendering, valid until the next swap (`SwapTexture`)."""
        return SwapTexture(self, self._fbos[1 - self._swap_state], self._swaps)

    def allocate(self, width: int, height: int, internal_format) -> None:
        """Allocate both buffers with same parameters."""
        self._fbos[0].allocate(width, height, internal_format)
        self._fbos[1].allocate(width, height, internal_format)

    def deallocate(self) -> None:
        """Deallocate both buffers."""
        self._fbos[0].deallocate()
        self._fbos[1].deallocate()

    def swap(self) -> None:
        """Swap buffers (back becomes front)."""
        self._swap_state = 1 - self._swap_state
        self._swaps += 1

    def begin(self) -> None:
        """Begin rendering to current buffer. Uses top-left origin."""
        self._fbos[self._swap_state].begin()

    def end(self) -> None:
        """End rendering to current buffer."""
        self._fbos[self._swap_state].end()

    def bind(self) -> None:
        """Bind current buffer's texture."""
        self._fbos[self._swap_state].bind()

    def unbind(self) -> None:
        """Unbind current buffer's texture."""
        self._fbos[self._swap_state].unbind()

    def clear(self, r: float = 0, g: float = 0, b: float = 0, a: float = 0.0) -> None:
        """Clear current buffer."""
        self._fbos[self._swap_state].clear(r, g, b, a)

    def clear_back(self, r: float = 0, g: float = 0, b: float = 0, a: float = 0.0) -> None:
        """Clear previous buffer."""
        self._fbos[1 - self._swap_state].clear(r, g, b, a)

    def set_wrap(self, wrap: int, border_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)) -> None:
        """Change wrap mode on both buffers at runtime (GL thread only)."""
        self._fbos[0].set_wrap(wrap, border_color)
        self._fbos[1].set_wrap(wrap, border_color)

    def clear_all(self, r: float = 0, g: float = 0, b: float = 0, a: float = 0.0) -> None:
        """Clear both buffers (useful for initialization/reset)."""
        self.clear(r, g, b, a)
        self.clear_back(r, g, b, a)