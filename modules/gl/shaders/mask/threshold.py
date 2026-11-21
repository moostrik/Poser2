"""Threshold: 1 where the input's red reaches the threshold, 0 below it."""

from OpenGL.GL import glUniform1f, glUseProgram
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
        src.bind(0)
        glUniform1f(self.get_uniform_loc("threshold"), threshold)
        draw_quad()
