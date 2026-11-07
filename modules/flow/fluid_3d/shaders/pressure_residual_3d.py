"""PressureResidual3D - the residual of the pressure's Poisson equation, for the multigrid's coarser levels."""
from __future__ import annotations

from OpenGL.GL import GL_R32F, glUniform1f, glUniform1i, glUniform3i, glUseProgram

from modules.gl import Texture3D, ComputeShader


class PressureResidual3D(ComputeShader):
    """r = div - L p on one level, its walls as the solvers take them."""

    WORKGROUP_SIZE_X = 8
    WORKGROUP_SIZE_Y = 8
    WORKGROUP_SIZE_Z = 4

    def use(self, pressure: Texture3D, divergence: Texture3D, obstacle: Texture3D, residual: Texture3D, cell: float,
            has_obstacles: bool, closed: tuple[bool, bool, bool]) -> None:
        if not self.allocated or not self.shader_program:
            return
        glUseProgram(self.shader_program)
        self.bind_texture_3d(0, divergence, "uDivergence")
        self.bind_texture_3d(1, obstacle, "uObstacle")
        self.bind_image_3d_read(0, pressure, GL_R32F)
        self.bind_image_3d_write(1, residual, GL_R32F)
        glUniform1f(self.get_uniform_loc("uAlpha"), 1.0 / (cell * cell))
        glUniform3i(self.get_uniform_loc("uSize"), pressure.width, pressure.height, pressure.depth)
        glUniform1i(self.get_uniform_loc("uHasObstacles"), int(has_obstacles))
        glUniform3i(self.get_uniform_loc("uClosed"), *(int(c) for c in closed))
        self.dispatch(pressure.width, pressure.height, pressure.depth)
