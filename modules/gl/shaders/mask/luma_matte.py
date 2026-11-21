"""LumaMatte: a matte from an image's brightness, its darkness made transparency."""

from OpenGL.GL import glUniform1f, glUseProgram
from ... import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)


class LumaMatte(Shader):
    """`src` with its alpha times 1 − `amount` + `amount` × its brightest channel (HSV's value); the color kept."""

    def use(self, src: Texture, amount: float) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("LumaMatte shader not allocated or shader program missing.")
            return
        if not src.allocated:
            logger.warning("LumaMatte shader: input texture not allocated.")
            return
        glUseProgram(self.shader_program)
        src.bind(0)
        glUniform1f(self.get_uniform_loc("amount"), amount)
        draw_quad()
