"""Glyph Arrows shader.

A velocity volume as lit 3D arrows on a grid, in one instanced draw.
"""

import numpy as np
from OpenGL.GL import (GL_TEXTURE0, GL_TEXTURE_3D, GL_TRIANGLES, GL_TRUE, glActiveTexture, glBindTexture,
                       glDrawArraysInstanced, glUniform1f, glUniform1i, glUniform3f, glUniform3i, glUniformMatrix4fv,
                       glUseProgram)
from modules.gl import Shader, Texture3D

VERTICES: int = 96          # the arrow's mesh: 8 sides of shaft, cone and its base


class GlyphArrows(Shader):
    """Draws an arrow per grid point along the volume's velocity; the depth test is the caller's."""

    def use(self, volume: Texture3D, view_projection: np.ndarray, box: tuple[float, float, float],
            grid: tuple[int, int, int], scale: float) -> None:
        """Draw the arrows.

        Args:
            volume: The velocity (xyz)
            view_projection: The box's space to clip space (4 × 4, row-major)
            box: The box's size, its width 1
            grid: The grid points across, down and deep
            scale: Multiplier on the velocity
        """
        if not self.allocated or not self.shader_program or not volume.allocated:
            return
        glUseProgram(self.shader_program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_3D, volume.tex_id)
        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniformMatrix4fv(self.get_uniform_loc("view_projection"), 1, GL_TRUE, view_projection.astype(np.float32))
        glUniform3f(self.get_uniform_loc("box"), *box)
        glUniform3i(self.get_uniform_loc("grid"), *grid)
        glUniform1f(self.get_uniform_loc("scale"), scale)
        glDrawArraysInstanced(GL_TRIANGLES, 0, VERTICES, grid[0] * grid[1] * grid[2])
        glBindTexture(GL_TEXTURE_3D, 0)
