"""Colorize: an image's hue and saturation drawn toward a color, its value kept (Photoshop's colorize)."""

from OpenGL.GL import glUniform1f, glUniform3f, glUseProgram
from modules.utils import Color
from ... import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)


class Colorize(Shader):
    """`src`'s hue moved toward `color`'s and its saturation raised toward it, by `strength` (0..1); its value kept."""

    def use(self, src: Texture, color: Color, strength: float = 0.5) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("Colorize shader not allocated or shader program missing.")
            return
        if not src.allocated:
            logger.warning("Colorize shader: input texture not allocated.")
            return

        glUseProgram(self.shader_program)
        src.bind(0)
        glUniform3f(self.get_uniform_loc("color"), color.r, color.g, color.b)
        glUniform1f(self.get_uniform_loc("strength"), strength)
        draw_quad()
