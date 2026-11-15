"""Levels: input black and white points and gamma, then output black and white points."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram
from ... import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)


class Levels(Shader):
    """`src` with `in_black`..`in_white` stretched to 0..1, raised to 1 / `gamma`, then laid onto
    `out_black`..`out_white`; every channel alike, alpha kept."""

    def use(self, src: Texture, in_black: float = 0.0, in_white: float = 1.0, gamma: float = 1.0,
            out_black: float = 0.0, out_white: float = 1.0) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("Levels shader not allocated or shader program missing.")
            return
        if not src.allocated:
            return
        glUseProgram(self.shader_program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, src.tex_id)
        glUniform1i(self.get_uniform_loc("src"), 0)
        glUniform1f(self.get_uniform_loc("inBlack"), in_black)
        glUniform1f(self.get_uniform_loc("inWhite"), in_white)
        glUniform1f(self.get_uniform_loc("gamma"), gamma)
        glUniform1f(self.get_uniform_loc("outBlack"), out_black)
        glUniform1f(self.get_uniform_loc("outWhite"), out_white)
        draw_quad()
