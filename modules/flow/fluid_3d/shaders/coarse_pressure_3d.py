"""CoarsePressure3D - the multigrid's coarsest level solved in one dispatch."""
from __future__ import annotations

from OpenGL.GL import GL_R32F, glDispatchCompute, glUniform1f, glUniform1i, glUniform3i, glUseProgram

from modules.gl import Texture3D, ComputeShader


class CoarsePressure3D(ComputeShader):
    """Red-black Gauss–Seidel sweeps of a level of at most `MAX_CELLS` cells by one workgroup in shared memory."""

    MAX_CELLS: int = 4096       # coarse_pressure_3d.comp's MAX_CELLS: two floats a cell, 32 KB of shared memory

    def fits(self, pressure: Texture3D) -> bool:
        return pressure.width * pressure.height * pressure.depth <= self.MAX_CELLS

    def solve(self, pressure: Texture3D, divergence: Texture3D, cell: float, sweeps: int,
              closed: tuple[bool, bool, bool]) -> None:
        """`sweeps` red-black sweeps on `pressure` in place (R32F), `divergence` its right-hand side."""
        if not self.allocated or not self.shader_program:
            return
        alpha = 1.0 / (cell * cell)
        glUseProgram(self.shader_program)
        self.bind_texture_3d(0, divergence, "uDivergence")
        self.bind_image_3d_readwrite(0, pressure, GL_R32F)
        glUniform1f(self.get_uniform_loc("uAlpha"), alpha)
        glUniform1f(self.get_uniform_loc("uBeta"), 1.0 / (6.0 * alpha))
        glUniform1i(self.get_uniform_loc("uSweeps"), sweeps)
        glUniform3i(self.get_uniform_loc("uSize"), pressure.width, pressure.height, pressure.depth)
        glUniform3i(self.get_uniform_loc("uClosed"), *(int(c) for c in closed))
        glDispatchCompute(1, 1, 1)
