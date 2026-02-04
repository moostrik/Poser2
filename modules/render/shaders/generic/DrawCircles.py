"""Draw multiple circles with antialiasing."""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad


class DrawCircles(Shader):
    """Draw up to 8 circles with individual positions, sizes, and colors."""

    def use(self, positions: list[tuple[float, float]], size: float, smooth: float,
            color: tuple[float, float, float, float], aspect_ratio: float = 1.0) -> None:
        """Draw circles.

        Args:
            positions: List of (x, y) tuples in normalized coordinates [0, 1]
            size: Circle radius in normalized coordinates (shared by all circles)
            smooth: Antialiasing smoothness in normalized coordinates (shared by all circles)
            color: (r, g, b, a) color tuple [0.0, 1.0] (shared by all circles)
            aspect_ratio: Width / height of rendering target for correct circle shape.
            Maximum 8 circles supported.
        """
        if not self.allocated or not self.shader_program:
            print("DrawCircles shader not allocated or shader program missing.")
            return

        num_circles = min(len(positions), 8)
        if num_circles == 0:
            return

        glUseProgram(self.shader_program)
        glUniform1i(self.get_uniform_loc("num_circles"), num_circles)
        glUniform1f(self.get_uniform_loc("aspect_ratio"), aspect_ratio)

        # Pack circle data into vec4 arrays
        circle_data = []
        color_data = []
        for i in range(num_circles):
            x, y = positions[i]
            circle_data.extend([x, y, size, smooth])
            color_data.extend(color)

        glUniform4fv(self.get_uniform_loc("circles"), num_circles, circle_data)
        glUniform4fv(self.get_uniform_loc("colors"), num_circles, color_data)

        draw_quad()
