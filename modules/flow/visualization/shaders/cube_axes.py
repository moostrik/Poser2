"""Cube Axes shader.

The box's x, y and z edges from its first corner, thicker and in the colormap's colors, so the 3D
views show which way each axis runs.
"""

import numpy as np
from OpenGL.GL import GL_TRIANGLES, GL_TRUE, GL_VIEWPORT, glDrawArrays, glGetIntegerv, glUniform1f, glUniform2f, glUniform3f, \
                      glUniformMatrix4fv, glUseProgram
from modules.gl import Shader

THICKNESS: float = 3.0          # pixels


class CubeAxes(Shader):
    """Draws the three axes as screen-wide quads; reads no vertex buffer."""

    def use(self, view_projection: np.ndarray, box: tuple[float, float, float]) -> None:
        """Draw the axes into the target bound.

        Args:
            view_projection: The box's space to clip space (4 × 4, row-major)
            box: The box's size, its width 1
        """
        if not self.allocated or not self.shader_program:
            return
        viewport = glGetIntegerv(GL_VIEWPORT)
        glUseProgram(self.shader_program)
        glUniformMatrix4fv(self.get_uniform_loc("view_projection"), 1, GL_TRUE, view_projection.astype(np.float32))
        glUniform3f(self.get_uniform_loc("box"), *box)
        glUniform2f(self.get_uniform_loc("resolution"), float(viewport[2]), float(viewport[3]))
        glUniform1f(self.get_uniform_loc("thickness"), THICKNESS)
        glDrawArrays(GL_TRIANGLES, 0, 18)
