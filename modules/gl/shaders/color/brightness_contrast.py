"""BrightnessContrast: contrast about middle grey, then brightness as a factor."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram
from ... import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)


class BrightnessContrast(Shader):
    """`src` with its contrast scaled about 0.5 and its brightness multiplied, clamped to 0..1."""

    def use(self, src: Texture, brightness: float, contrast: float) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("BrightnessContrast shader not allocated or shader program missing.")
            return
        if not src.allocated:
            logger.warning("BrightnessContrast shader: input texture not allocated.")
            return

        glUseProgram(self.shader_program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, src.tex_id)
        glUniform1i(self.get_uniform_loc("src"), 0)
        glUniform1f(self.get_uniform_loc("brightness"), brightness)
        glUniform1f(self.get_uniform_loc("contrast"), contrast)
        draw_quad()
