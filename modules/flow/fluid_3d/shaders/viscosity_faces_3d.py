"""ViscosityFaces3D - the viscosity's Jacobi iterations on the velocity's faces."""
from __future__ import annotations

import math

from OpenGL.GL import glUniform1f, glUniform1i, glUseProgram

from modules.gl import Texture3D, ComputeShader
from .face_binding import bind_faces, bind_face_writes

class ViscosityFaces3D(ComputeShader):
    """Each component's 7-point Laplacian over its own faces, iterated in shared memory a tile at a time."""

    WORKGROUP_SIZE_X = 16
    WORKGROUP_SIZE_Y = 16
    WORKGROUP_SIZE_Z = 2
    ITERATIONS_PER_DISPATCH = 5

    def use(self, source: Texture3D, result: Texture3D, obstacle: Texture3D, cells: tuple[int, int, int],
            cell: float, viscosity_dt: float, iterations: int, closed: tuple[bool, bool, bool], wall_keep: float,
            has_obstacles: bool) -> None:
        if not self.allocated or not self.shader_program:
            return
        alpha = 1.0 / (cell * cell)
        gamma = 1.0 / max(viscosity_dt, 1e-6)
        glUseProgram(self.shader_program)
        bind_faces(self, source, cells, closed, has_obstacles, obstacle)
        bind_face_writes(self, result)
        glUniform1f(self.get_uniform_loc('uAlpha'), alpha)
        glUniform1f(self.get_uniform_loc('uGamma'), gamma)
        glUniform1f(self.get_uniform_loc('uBeta'), 1.0 / (6.0 * alpha + gamma))
        glUniform1f(self.get_uniform_loc('uWallKeep'), wall_keep)
        glUniform1i(self.get_uniform_loc('uIterations'), iterations)
        self.dispatch(cells[0] + 1, cells[1] + 1, cells[2] + 1)

    def dispatches(self, total_iterations: int) -> list[int]:
        """The iterations of each dispatch, `ITERATIONS_PER_DISPATCH` at most."""
        count = math.ceil(total_iterations / self.ITERATIONS_PER_DISPATCH)
        return [min(self.ITERATIONS_PER_DISPATCH, total_iterations - i * self.ITERATIONS_PER_DISPATCH) for i in range(count)]
