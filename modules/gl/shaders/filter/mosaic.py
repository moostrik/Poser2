"""Mosaic: the image in square blocks, each the color at its center (pixelate)."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram
from ... import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)


class Mosaic(Shader):
    """`src` in blocks of `block` of its pixels, each the color sampled at its center."""

    def use(self, src: Texture, block: float = 16.0) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("Mosaic shader not allocated or shader program missing.")
            return
        if not src.allocated:
            return
        glUseProgram(self.shader_program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, src.tex_id)
        glUniform1i(self.get_uniform_loc("src"), 0)
        glUniform1f(self.get_uniform_loc("block"), max(block, 1.0))
        draw_quad()
