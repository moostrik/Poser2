"""DrawBox: a solid-color rectangle, the box behind a text."""

from OpenGL.GL import glUniform4f, glUseProgram
from modules.utils import Color, Rect
from .. import Shader, draw_quad


class DrawBox(Shader):
    """A solid-color rectangle: `rect` in 0..1 of the target, origin top-left, in `color`."""

    def use(self, rect: Rect, color: Color) -> None:
        if not self.allocated or not self.shader_program:
            return

        glUseProgram(self.shader_program)
        glUniform4f(self.get_uniform_loc("rect"), rect.x, rect.y, rect.width, rect.height)
        glUniform4f(self.get_uniform_loc("color"), *color)

        draw_quad()
