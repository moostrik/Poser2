"""Displace: an image's pixels moved by a displacement map."""

from OpenGL.GL import glUniform1f, glUseProgram
from ... import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)


class Displace(Shader):
    """`src` read where `map` sends each pixel: its red and green, 0.5 for no move, times `amount` in 0..1 of the
    image across and down."""

    def use(self, src: Texture, map: Texture, amount: float = 0.05) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("Displace shader not allocated or shader program missing.")
            return
        if not src.allocated or not map.allocated:
            return
        glUseProgram(self.shader_program)
        src.bind(0)
        map.bind(1)
        glUniform1f(self.get_uniform_loc("amount"), amount)
        draw_quad()
