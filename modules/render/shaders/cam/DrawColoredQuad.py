"""Draw colored quad shader - fills viewport with solid color."""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl.Utils import viewport_rect


class DrawColoredQuad(Shader):
    """Draw a filled quad with specified color and position."""

    def use(self, x: float, y: float, w: float, h: float, r: float, g: float, b: float, a: float) -> None:
        """Draw a colored quad at specified position.

        Args:
            x: X position (top-left, pixel coordinates)
            y: Y position (top-left, pixel coordinates)
            w: Width in pixels
            h: Height in pixels
            r, g, b, a: Color components (0.0-1.0)
        """
        if not self.allocated or not self.shader_program:
            print("DrawColoredQuad shader not allocated or shader program missing.")
            return

        glUseProgram(self.shader_program)

        # Set viewport to position the quad
        viewport_rect(x, y, w, h)

        # Set color uniform
        glUniform4f(self.get_uniform_loc("color"), r, g, b, a)

        # Draw fullscreen quad (fills viewport)
        draw_quad()

        glUseProgram(0)
