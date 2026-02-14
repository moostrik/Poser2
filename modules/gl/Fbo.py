from OpenGL.GL import * # type: ignore
from modules.gl.Texture import Texture

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
        """Current write buffer."""
        return self._fbos[self._swap_state]

    @property
    def back_texture(self) -> Texture:
        """Previous buffer for reading in ping-pong rendering."""
        return self._fbos[1 - self._swap_state]

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

    def clear_all(self, r: float = 0, g: float = 0, b: float = 0, a: float = 0.0) -> None:
        """Clear both buffers (useful for initialization/reset)."""
        self.clear(r, g, b, a)
        self.clear_back(r, g, b, a)