"""Blit: a texture drawn into the target, or a part of it into a part of the target."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUniform4f, glUseProgram
from modules.utils import Rect
from .. import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)

WHOLE: Rect = Rect(0.0, 0.0, 1.0, 1.0)


class Blit(Shader):
    """A texture's `src` drawn at `dst` in the target, both 0..1 with the origin top-left; a `src` with a negative
    width or height draws it mirrored on that axis."""

    def use(self, tex: Texture, dst: Rect | None = None, src: Rect | None = None, opacity: float = 1.0) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("Blit shader not allocated or shader program missing.")
            return
        if not tex.allocated:
            return
        d: Rect = dst or WHOLE
        s: Rect = src or WHOLE

        glUseProgram(self.shader_program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex.tex_id)
        glUniform1i(self.get_uniform_loc("tex"), 0)
        glUniform1f(self.get_uniform_loc("opacity"), opacity)
        glUniform4f(self.get_uniform_loc("dst"), d.x, d.y, d.width, d.height)
        glUniform4f(self.get_uniform_loc("src"), s.x, s.y, s.width, s.height)
        draw_quad()
