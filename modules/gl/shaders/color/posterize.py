"""Posterize: every channel rounded to a number of levels."""

from OpenGL.GL import glUniform1f, glUseProgram
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
        src.bind(0)
        glUniform1f(self.get_uniform_loc("levels"), float(max(levels, 2)))
        draw_quad()
