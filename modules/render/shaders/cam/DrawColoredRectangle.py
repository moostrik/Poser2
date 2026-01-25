"""Draw colored rectangle shader - positioned rectangle with solid color."""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad


class DrawColoredRectangle(Shader):
    """Draw a filled rectangle with specified color and position."""

    def use(self, x: float, y: float, w: float, h: float, r: float, g: float, b: float, a: float) -> None:
        """Draw a colored rectangle at specified position.

        Args:
            x: X position (normalized 0..1, left edge)
            y: Y position (normalized 0..1, top edge)
            w: Width (normalized 0..1)
            h: Height (normalized 0..1)
            r, g, b, a: Color components (0.0-1.0)
        """
        if not self.allocated or not self.shader_program:
            print("DrawColoredRectangle shader not allocated or shader program missing.")
            return

        glUseProgram(self.shader_program)
        glUniform4f(self.get_uniform_loc("rect"), x, y, w, h)
        glUniform4f(self.get_uniform_loc("color"), r, g, b, a)
        draw_quad()
