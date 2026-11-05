"""Log Grid shader.

Draws the lines of a log axis: a line in each lane at every height given, and optionally the middle.
"""

from OpenGL.GL import glUniform1i, glUniform4f, glUseProgram
from modules.gl import Shader, draw_quad


class LogGrid(Shader):
    """Faint grey lines on transparent."""

    def use(self, lanes: int, fractions: list[float], center: bool) -> None:
        """Draw the lines.

        Args:
            lanes: Lanes, top to bottom
            fractions: Up to four heights in a lane, 0 bottom, 1 top
            center: A line down the middle
        """
        if not self.allocated or not self.shader_program:
            return

        padded: list[float] = (list(fractions) + [0.0] * 4)[:4]
        glUseProgram(self.shader_program)
        glUniform1i(self.get_uniform_loc("lanes"), lanes)
        glUniform4f(self.get_uniform_loc("fractions"), *padded)
        glUniform1i(self.get_uniform_loc("count"), min(len(fractions), 4))
        glUniform1i(self.get_uniform_loc("center"), 1 if center else 0)

        draw_quad()
