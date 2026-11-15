"""Blit: a texture drawn into the target, or a part of it into a part of the target."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUniform4f, glUseProgram
from modules.utils import Rect
from ... import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)

WHOLE: Rect = Rect(0.0, 0.0, 1.0, 1.0)


class Blit(Shader):
    """`src`'s `src_rect` drawn at `dst_rect` in the target, both 0..1 with the origin top-left; a `src_rect` with a
    negative width or height draws it mirrored on that axis."""

    def use(self, src: Texture, dst_rect: Rect | None = None, src_rect: Rect | None = None, opacity: float = 1.0) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("Blit shader not allocated or shader program missing.")
            return
        if not src.allocated:
            return
        d: Rect = dst_rect or WHOLE
        s: Rect = src_rect or WHOLE

        glUseProgram(self.shader_program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, src.tex_id)
        glUniform1i(self.get_uniform_loc("src"), 0)
        glUniform1f(self.get_uniform_loc("opacity"), opacity)
        glUniform4f(self.get_uniform_loc("dstRect"), d.x, d.y, d.width, d.height)
        glUniform4f(self.get_uniform_loc("srcRect"), s.x, s.y, s.width, s.height)
        draw_quad()
