"""Sobel: the Sobel edge magnitude of the input's luminance, as lines past a threshold."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram
from ... import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)


class Sobel(Shader):
    """`src`'s edges: the Sobel magnitude times `strength`, smoothly stepped at `threshold`; `invert` draws dark lines
    on white."""

    def use(self, src: Texture, threshold: float = 0.1, strength: float = 1.0, invert: bool = False) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("Sobel shader not allocated or shader program missing.")
            return
        if not src.allocated:
            logger.warning("Sobel shader: input texture not allocated.")
            return

        glUseProgram(self.shader_program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, src.tex_id)
        glUniform1i(self.get_uniform_loc("src"), 0)
        glUniform1f(self.get_uniform_loc("threshold"), threshold)
        glUniform1f(self.get_uniform_loc("strength"), strength)
        glUniform1i(self.get_uniform_loc("invert"), 1 if invert else 0)
        draw_quad()
