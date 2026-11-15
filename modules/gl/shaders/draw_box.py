"""DrawBox: a solid-color rectangle placed in pixels, the box behind a text."""

from OpenGL.GL import glUniform2f, glUniform4f, glUseProgram
from .. import Shader, draw_quad


class DrawBox(Shader):
    """A solid-color rectangle, placed in the target's pixels."""

    def use(self, rect: tuple[float, float, float, float],
            color: tuple[float, float, float, float],
            screen_size: tuple[float, float]) -> None:
        """Render a solid-color rectangle.

        Args:
            rect: (x, y, width, height) in screen pixels
            color: (r, g, b, a) fill color
            screen_size: (width, height) of render target
        """
        if not self.allocated or not self.shader_program:
            return

        glUseProgram(self.shader_program)

        glUniform2f(self.get_uniform_loc("screenSize"), *screen_size)
        glUniform4f(self.get_uniform_loc("rect"), *rect)
        glUniform4f(self.get_uniform_loc("color"), *color)

        draw_quad()
