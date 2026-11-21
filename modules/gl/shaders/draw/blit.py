"""Blit: a texture drawn into the target, or a part of it into a part of the target."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform2fv, glUniform4f, glUseProgram
from modules.utils import Quad, Rect
from ... import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)

WHOLE: Rect = Rect(0.0, 0.0, 1.0, 1.0)


class Blit(Shader):
    """`src`'s part `src_part` drawn at `dst_rect` in the target, all 0..1 with the origin top-left. `src_part` is a
    `Rect`, mirrored on an axis by a negative width or height, or a `Quad` of four corners, a turned part."""

    def use(self, src: Texture, dst_rect: Rect | None = None, src_part: Rect | Quad | None = None, opacity: float = 1.0) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("Blit shader not allocated or shader program missing.")
            return
        if not src.allocated:
            return
        d: Rect = dst_rect or WHOLE
        part: Rect | Quad = src_part or WHOLE
        quad: Quad = part.corners() if isinstance(part, Rect) else part

        glUseProgram(self.shader_program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, src.tex_id)
        glUniform1f(self.get_uniform_loc("opacity"), opacity)
        glUniform4f(self.get_uniform_loc("dstRect"), d.x, d.y, d.width, d.height)
        glUniform2fv(self.get_uniform_loc("srcQuad"), 4, [*quad.top_left, *quad.top_right, *quad.bottom_right, *quad.bottom_left])
        draw_quad()
