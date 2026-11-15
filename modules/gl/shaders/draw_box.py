"""DrawBox: a solid-color rectangle, the box behind a text."""

from OpenGL.GL import glUniform4f, glUseProgram
from .. import Shader, draw_quad


class DrawBox(Shader):
    """A solid-color rectangle, placed in 0..1 of the target, origin top-left."""

    def use(self, rect: tuple[float, float, float, float], color: tuple[float, float, float, float]) -> None:
        """`rect` x, y, width, height in 0..1 of the target, origin top-left; `color` r, g, b, a."""
        if not self.allocated or not self.shader_program:
            return

        glUseProgram(self.shader_program)
        glUniform4f(self.get_uniform_loc("rect"), *rect)
        glUniform4f(self.get_uniform_loc("color"), *color)

        draw_quad()
