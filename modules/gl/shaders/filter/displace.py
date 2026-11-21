"""Displace: an image's pixels moved by a displacement map."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUseProgram
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
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, src.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, map.tex_id)
        glUniform1f(self.get_uniform_loc("amount"), amount)
        draw_quad()
