"""BrightnessContrast: contrast about middle grey, then brightness as a factor."""

from OpenGL.GL import glUniform1f, glUseProgram
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
        src.bind(0)
        glUniform1f(self.get_uniform_loc("brightness"), brightness)
        glUniform1f(self.get_uniform_loc("contrast"), contrast)
        draw_quad()
