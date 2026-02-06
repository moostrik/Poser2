"""Box rendering shader.

Renders solid-color rectangles for text backgrounds.
"""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad


class BoxShader(Shader):
    """Shader for rendering solid-color background boxes."""

    def use(self, box_rect: tuple[float, float, float, float],
            box_color: tuple[float, float, float, float],
            screen_size: tuple[float, float]) -> None:
        """Render a solid-color rectangle.

        Args:
            box_rect: (x, y, width, height) in screen pixels
            box_color: (r, g, b, a) fill color
            screen_size: (width, height) of render target
        """
        if not self.allocated or not self.shader_program:
            return

        glUseProgram(self.shader_program)

        # Set uniforms
        glUniform2f(self.get_uniform_loc("screen_size"), *screen_size)
        glUniform4f(self.get_uniform_loc("box_rect"), *box_rect)
        glUniform4f(self.get_uniform_loc("box_color"), *box_color)

        # Draw quad
        draw_quad()
