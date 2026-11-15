"""Posterize: every channel rounded to a number of levels."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram
from ... import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)


class Posterize(Shader):
    """`src`'s RGB in `levels` steps from 0 to 1, alpha kept."""

    def use(self, src: Texture, levels: int = 4) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("Posterize shader not allocated or shader program missing.")
            return
        if not src.allocated:
            return
        glUseProgram(self.shader_program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, src.tex_id)
        glUniform1i(self.get_uniform_loc("src"), 0)
        glUniform1f(self.get_uniform_loc("levels"), float(max(levels, 2)))
        draw_quad()
