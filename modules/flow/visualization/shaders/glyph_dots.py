"""Glyph Dots shader.

A scalar volume as 3D dots on a grid, sized and colored by the value, in one draw of points.
"""

import numpy as np
from OpenGL.GL import (GL_POINTS, GL_PROGRAM_POINT_SIZE, GL_TEXTURE0, GL_TEXTURE_3D, GL_TRUE, glActiveTexture,
                       glBindTexture, glDisable, glDrawArrays, glEnable, glUniform1f, glUniform1i, glUniform3f,
                       glUniform3i, glUniformMatrix4fv, glUseProgram)
from modules.gl import Shader, Texture3D


class GlyphDots(Shader):
    """Draws a dot per grid point of the volume; the depth test is the caller's."""

    def use(self, volume: Texture3D, view_projection: np.ndarray, box: tuple[float, float, float],
            grid: tuple[int, int, int], scale: float, channel: int, pixels_per_unit: float) -> None:
        """Draw the dots.

        Args:
            volume: The volume
            view_projection: The box's space to clip space (4 × 4, row-major)
            box: The box's size, its width 1
            grid: The grid points across, down and deep
            scale: Multiplier on the values
            channel: 0 to 3, or 4 for all
            pixels_per_unit: Pixels per box unit at depth 1: the viewport's height times half the projection's [1][1]
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
        glUniform1i(self.get_uniform_loc("channel"), channel)
        glUniform1f(self.get_uniform_loc("pixels_per_unit"), pixels_per_unit)
        glEnable(GL_PROGRAM_POINT_SIZE)
        glDrawArrays(GL_POINTS, 0, grid[0] * grid[1] * grid[2])
        glDisable(GL_PROGRAM_POINT_SIZE)
        glBindTexture(GL_TEXTURE_3D, 0)
