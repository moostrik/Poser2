"""Colormap Key shader.

Draws the colormap's key: a bar per channel from -1 through 0 to 1, in its colors.
"""

from OpenGL.GL import glUniform1f, glUniform1i, glUniform4f, glUseProgram
from modules.gl import Shader, draw_quad


class ColormapKey(Shader):
    """Color bars on transparent, in a box."""

    def use(self, box: tuple[float, float, float, float], channel: int, channels: int, bar_height: float, bar_gap: float) -> None:
        """Draw the key.

        Args:
            box: x0, y0, x1, y1 in pixels from the bottom left
            channel: 0 to 3 for that channel's bar, 4 for each of x, y and z the texture holds
            channels: The channels the texture holds, x first, 1 to 4
            bar_height: A bar's height in pixels
            bar_gap: The gap between bars in pixels
        """
        if not self.allocated or not self.shader_program:
            return

        glUseProgram(self.shader_program)
        glUniform4f(self.get_uniform_loc("box"), *box)
        glUniform1i(self.get_uniform_loc("channel"), channel)
        glUniform1i(self.get_uniform_loc("channels"), channels)
        glUniform1f(self.get_uniform_loc("bar_height"), bar_height)
        glUniform1f(self.get_uniform_loc("bar_gap"), bar_gap)

        draw_quad()
