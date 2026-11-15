"""Median: each pixel the median of its 3 x 3 neighborhood, per channel."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1i, glUseProgram
from ... import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)


class Median(Shader):
    """`src` with every channel the median of its 3 x 3 neighbors (the edges clamped), against salt-and-pepper
    noise without a blur's soft edges."""

    def use(self, src: Texture) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("Median shader not allocated or shader program missing.")
            return
        if not src.allocated:
            return
        glUseProgram(self.shader_program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, src.tex_id)
        glUniform1i(self.get_uniform_loc("src"), 0)
        draw_quad()
