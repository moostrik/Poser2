"""Cube Edges shader.

The box's twelve edges as lines, for orientation in the 3D views.
"""

import numpy as np
from OpenGL.GL import GL_LINES, GL_TRUE, glDrawArrays, glUniform3f, glUniform4f, glUniformMatrix4fv, glUseProgram
from modules.gl import Shader


class CubeEdges(Shader):
    """Draws the box's edges in one color; reads no vertex buffer."""

    def use(self, view_projection: np.ndarray, box: tuple[float, float, float],
            color: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)) -> None:
        """Draw the edges.

        Args:
            view_projection: The box's space to clip space (4 × 4, row-major)
            box: The box's size, its width 1
            color: The lines' color
        """
        if not self.allocated or not self.shader_program:
            return
        glUseProgram(self.shader_program)
        glUniformMatrix4fv(self.get_uniform_loc("view_projection"), 1, GL_TRUE, view_projection.astype(np.float32))
        glUniform3f(self.get_uniform_loc("box"), *box)
        glUniform4f(self.get_uniform_loc("color"), *color)
        glDrawArrays(GL_LINES, 0, 24)
