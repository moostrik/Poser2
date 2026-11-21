"""Erode: the morphological erosion of the input's red, a 3 x 3 minimum."""

from OpenGL.GL import glUniform1f, glUseProgram
from ... import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)


class Erode(Shader):
    """`src`'s red, each pixel the smallest of its 3 x 3 neighbors `radius` pixels apart."""

    def use(self, src: Texture, radius: float = 1.0) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("Erode shader not allocated or shader program missing.")
            return
        if not src.allocated:
            logger.warning("Erode shader: input texture not allocated.")
            return

        glUseProgram(self.shader_program)
        src.bind(0)
        glUniform1f(self.get_uniform_loc("radius"), radius)
        draw_quad()
