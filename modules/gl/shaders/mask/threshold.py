"""Threshold: 1 where the input's red reaches the threshold, 0 below it."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUseProgram
from ... import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)


class Threshold(Shader):
    """`src`'s red stepped at `threshold`, written to RGB."""

    def use(self, src: Texture, threshold: float = 0.5) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("Threshold shader not allocated or shader program missing.")
            return
        if not src.allocated:
            logger.warning("Threshold shader: input texture not allocated.")
            return

        glUseProgram(self.shader_program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, src.tex_id)
        glUniform1f(self.get_uniform_loc("threshold"), threshold)
        draw_quad()
