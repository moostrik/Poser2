"""Bilateral: a blur that keeps edges, each neighbor weighted by its distance and by how alike its color is."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram
from ... import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)


class Bilateral(Shader):
    """`src` averaged over a square of `radius` pixels each side (Tomasi and Manduchi 1998): each neighbor weighted
    by a Gaussian of its distance (`sigma_space`, pixels) times a Gaussian of its color's difference from the center's
    (`sigma_color`, 0..1)."""

    def use(self, src: Texture, radius: int = 3, sigma_space: float = 2.0, sigma_color: float = 0.1) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("Bilateral shader not allocated or shader program missing.")
            return
        if not src.allocated:
            return
        glUseProgram(self.shader_program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, src.tex_id)
        glUniform1i(self.get_uniform_loc("src"), 0)
        glUniform1i(self.get_uniform_loc("radius"), max(radius, 0))
        glUniform1f(self.get_uniform_loc("sigmaSpace"), max(sigma_space, 1e-4))
        glUniform1f(self.get_uniform_loc("sigmaColor"), max(sigma_color, 1e-4))
        draw_quad()
