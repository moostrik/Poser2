"""StrokeRect: a rectangle's outline."""

from OpenGL.GL import glUniform2f, glUniform4f, glUseProgram
from modules.utils import Color, Rect
from ... import Shader, draw_quad

import logging
logger = logging.getLogger(__name__)


class StrokeRect(Shader):
    """A rectangle's outline: `rect` in 0..1 of the target, origin top-left, in `color`; the line widths are parts of
    the rectangle's width (the vertical lines) and height (the horizontal ones)."""

    def use(self, rect: Rect, color: Color, line_width_x: float = 0.01, line_width_y: float = 0.01) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("StrokeRect shader not allocated or shader program missing.")
            return

        glUseProgram(self.shader_program)
        glUniform4f(self.get_uniform_loc("rect"), rect.x, rect.y, rect.width, rect.height)
        glUniform4f(self.get_uniform_loc("color"), *color)
        glUniform2f(self.get_uniform_loc("lineWidth"), line_width_x, line_width_y)
        draw_quad()
