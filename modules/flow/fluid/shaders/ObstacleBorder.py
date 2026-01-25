"""Obstacle border shader - creates border mask for fluid simulation."""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad


class ObstacleBorder(Shader):
    """Creates obstacle border mask (1 at edges, 0 inside)."""

    def use(self, width: int, height: int, border: float = 1.0) -> None:
        """Draw obstacle border mask.

        Args:
            width: Texture width in pixels
            height: Texture height in pixels
            border: Border width in pixels
        """
        if not self.allocated or not self.shader_program:
            print("ObstacleBorder shader not allocated or shader program missing.")
            return

        glUseProgram(self.shader_program)

        glUniform2f(self.get_uniform_loc("uResolution"), float(width), float(height))
        glUniform1f(self.get_uniform_loc("uBorder"), border)

        draw_quad()
