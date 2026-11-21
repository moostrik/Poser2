"""The objects of one GL context: the quad its draws run on, the shader a texture draws with, and its blend state.

As Cinder's `gl::Context`: one per GL context, made once that context is up and freed before it closes. A window's
context does not share its vertex arrays with another, so each holds its own quad; `context()` is the one made
current, which `Texture.draw` and `Text` draw through.
"""

from __future__ import annotations

import ctypes
import logging
from typing import TYPE_CHECKING

import numpy as np
from OpenGL.GL import (GL_ARRAY_BUFFER, GL_FALSE, GL_FLOAT, GL_STATIC_DRAW, GL_TRIANGLE_FAN, glBindBuffer,
                       glBindVertexArray, glBufferData, glDeleteBuffers, glDeleteVertexArrays, glDrawArrays,
                       glEnableVertexAttribArray, glGenBuffers, glGenVertexArrays, glVertexAttribPointer)

from . import style
from .style import StyleStack

if TYPE_CHECKING:
    from ..shaders import Blit

logger = logging.getLogger(__name__)


def draw_quad() -> None:
    """Draw the full-screen quad of the context made current; its vertex array is bound while it is current."""
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4)


class Context:
    """One GL context's quad, `Blit` and blend state. `allocate` and `make_current` while that context is current."""

    def __init__(self) -> None:
        self.style: StyleStack = StyleStack()
        self._quad_vao: int = 0
        self._quad_vbo: int = 0
        self._bound_vao: int = 0
        self._blit: Blit | None = None

    @property
    def allocated(self) -> bool:
        return self._quad_vao != 0

    @property
    def blit(self) -> Blit:
        """The shader a texture draws with (`Texture.draw`)."""
        assert self._blit is not None, 'gl.Context used before allocate()'
        return self._blit

    def allocate(self) -> None:
        """The quad, the `Blit` and GL's default blend state, in the context current now; and this made current."""
        from ..shaders import Blit             # the shaders import the textures this module is imported with
        # Two attributes per corner: the position (-1..1) and the texture coordinate (0..1), bottom-left first
        vertices = np.array([-1.0, -1.0, 0.0, 0.0,
                             1.0, -1.0, 1.0, 0.0,
                             1.0,  1.0, 1.0, 1.0,
                            -1.0,  1.0, 0.0, 1.0], dtype=np.float32)
        self._quad_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._quad_vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        self._quad_vao = glGenVertexArrays(1)
        glBindVertexArray(self._quad_vao)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(2 * 4))
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        self._bound_vao = self._quad_vao
        self.make_current()
        self.style.reset()
        self._blit = Blit()
        self._blit.allocate()

    def deallocate(self) -> None:
        """Free what `allocate` made, in this context, current now."""
        global _current
        if self._blit is not None:
            self._blit.deallocate()
            self._blit = None
        if self._quad_vao:
            glBindVertexArray(0)
            glDeleteVertexArrays(1, [self._quad_vao])
            glDeleteBuffers(1, [self._quad_vbo])
        self._quad_vao = self._quad_vbo = self._bound_vao = 0
        if _current is self:
            _current = None

    def make_current(self) -> None:
        """This context's objects for the draws that follow, its GL context made current by the caller: the quad
        bound, its blend state the one `Style` works on."""
        global _current
        _current = self
        style.use(self.style)
        self.bind_quad()

    def bind_quad(self) -> None:
        """The quad `draw_quad` draws, bound."""
        self.bind_vertex_array(self._quad_vao)

    def bind_vertex_array(self, vao: int) -> int:
        """Bind `vao`, and return the one bound before, as this context keeps it (no `glGet*`)."""
        before: int = self._bound_vao
        glBindVertexArray(vao)
        self._bound_vao = vao
        return before


_current: Context | None = None


def context() -> Context:
    """The context made current (`Context.make_current`)."""
    if _current is None:
        raise RuntimeError('no gl.Context made current')
    return _current
