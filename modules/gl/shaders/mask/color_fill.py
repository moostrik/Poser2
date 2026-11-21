"""ColorFill: one color, the input's red as its alpha."""

from OpenGL.GL import glUniform4f, glUseProgram
from modules.utils import Color
from ... import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)


class ColorFill(Shader):
    """`color` filled through the mask `src`: its RGB, the mask's red as alpha."""

    def use(self, src: Texture, color: Color) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("ColorFill shader not allocated or shader program missing.")
            return
        if not src.allocated:
            logger.warning("ColorFill shader: input texture not allocated.")
            return

        glUseProgram(self.shader_program)
        src.bind(0)
        glUniform4f(self.get_uniform_loc("color"), *color)
        draw_quad()
