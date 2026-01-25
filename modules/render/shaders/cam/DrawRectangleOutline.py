"""Draw rectangle outline shader - draws rectangle border with specified line width."""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad


class DrawRectangleOutline(Shader):
    """Draw a rectangle outline with specified color, position, and line width."""

    def use(self, x: float, y: float, w: float, h: float, r: float, g: float, b: float, a: float, line_width_x: float = 0.01, line_width_y: float = 0.01) -> None:
        """Draw a rectangle outline at specified position.

        Args:
            x: X position (normalized 0..1, left edge)
            y: Y position (normalized 0..1, top edge)
            w: Width (normalized 0..1)
            h: Height (normalized 0..1)
            r, g, b, a: Color components (0.0-1.0)
            line_width_x: Line width for vertical lines in normalized coordinates
            line_width_y: Line width for horizontal lines in normalized coordinates
        """
        if not self.allocated or not self.shader_program:
            print("DrawRectangleOutline shader not allocated or shader program missing.")
            return

        glUseProgram(self.shader_program)
        glUniform4f(self.get_uniform_loc("rect"), x, y, w, h)
        glUniform4f(self.get_uniform_loc("color"), r, g, b, a)
        glUniform2f(self.get_uniform_loc("lineWidth"), line_width_x, line_width_y)
        draw_quad()
