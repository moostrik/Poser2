"""Vignette: the image darkened toward its edges."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram
from ... import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)


class Vignette(Shader):
    """`src` darkened by `amount` (0..1) from `radius` (0..1 of the half diagonal) outward over `softness`, round in
    the image's pixels; alpha kept."""

    def use(self, src: Texture, amount: float = 0.5, radius: float = 0.75, softness: float = 0.45) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("Vignette shader not allocated or shader program missing.")
            return
        if not src.allocated:
            return
        glUseProgram(self.shader_program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, src.tex_id)
        glUniform1i(self.get_uniform_loc("src"), 0)
        glUniform1f(self.get_uniform_loc("amount"), amount)
        glUniform1f(self.get_uniform_loc("radius"), radius)
        glUniform1f(self.get_uniform_loc("softness"), max(softness, 1e-4))
        draw_quad()
