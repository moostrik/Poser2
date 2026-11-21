"""WeightedSum: two images, each times its weight, summed: `a * a_weight + b * b_weight`."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUseProgram
from ... import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)


class WeightedSum(Shader):
    """`a` times `a_weight` plus `b` times `b_weight`, every channel; without `b`, `a` scaled alone."""

    def use(self, a: Texture, a_weight: float, b: Texture | None = None, b_weight: float = 0.0) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("WeightedSum shader not allocated or shader program missing.")
            return
        if not a.allocated or (b is not None and not b.allocated):
            logger.warning("WeightedSum shader: input textures not allocated.")
            return

        glUseProgram(self.shader_program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, a.tex_id)
        glUniform1f(self.get_uniform_loc("aWeight"), a_weight)
        if b is not None:
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, b.tex_id)
        glUniform1f(self.get_uniform_loc("bWeight"), b_weight if b is not None else 0.0)
        draw_quad()
