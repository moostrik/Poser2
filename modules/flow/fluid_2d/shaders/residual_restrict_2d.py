"""ResidualRestrict2D - the 2D multigrid's residual restricted to the next coarser level in one pass."""
from __future__ import annotations

from OpenGL.GL import GL_R32F, glUniform1i, glUniform2f, glUniform2i, glUseProgram

from modules.gl import Texture, ComputeShader


class ResidualRestrict2D(ComputeShader):
    """Each coarse cell the mean of its fine cells' residuals, div - L p, worked out where they are read."""

    WORKGROUP_SIZE_X = 16
    WORKGROUP_SIZE_Y = 16
    WORKGROUP_SIZE_Z = 1

    def use(self, pressure: Texture, divergence: Texture, obstacle: Texture, coarse: Texture, cell: tuple[float, float],
            factor: tuple[int, int], has_obstacles: bool, closed: bool) -> None:
        """The fine level's (`pressure`, `divergence`, `cell`) residual restricted into `coarse` (R32F)."""
        if not self.allocated or not self.shader_program:
            return
        glUseProgram(self.shader_program)
        self.bind_texture(0, divergence, "uDivergence")
        self.bind_texture(1, obstacle, "uObstacle")
        self.bind_image_read(0, pressure, GL_R32F)
        self.bind_image_write(1, coarse, GL_R32F)
        glUniform2f(self.get_uniform_loc("uAlpha"), 1.0 / (cell[0] * cell[0]), 1.0 / (cell[1] * cell[1]))
        glUniform2i(self.get_uniform_loc("uFactor"), *factor)
        glUniform2i(self.get_uniform_loc("uSize"), pressure.width, pressure.height)
        glUniform2i(self.get_uniform_loc("uCoarseSize"), coarse.width, coarse.height)
        glUniform1i(self.get_uniform_loc("uHasObstacles"), int(has_obstacles))
        glUniform1i(self.get_uniform_loc("uClosed"), int(closed))
        self.dispatch(coarse.width, coarse.height)
