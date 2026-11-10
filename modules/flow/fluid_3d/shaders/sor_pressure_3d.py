"""SorPressure3D - the pressure by red-black Gauss–Seidel with over-relaxation, in place."""
from __future__ import annotations

from OpenGL.GL import GL_R32F, glUniform1f, glUniform1i, glUniform3f, glUniform3i, glUseProgram

from modules.gl import Texture3D, ComputeShader


class SorPressure3D(ComputeShader):
    """Red-black SOR on the pressure: each sweep a red and a black half, each color from the other's newest values."""

    WORKGROUP_SIZE_X = 8
    WORKGROUP_SIZE_Y = 8
    WORKGROUP_SIZE_Z = 4

    def solve(self, pressure: Texture3D, divergence: Texture3D, obstacle: Texture3D, cell: tuple[float, float, float], sweeps: int,
              omega: float, has_obstacles: bool = True, closed: tuple[bool, bool, bool] = (True, True, True)) -> None:
        """`sweeps` red-black sweeps on `pressure` in place, warm from its last values.

        Args:
            pressure: The pressure (R32F), read and written
            divergence: The velocity's divergence (R16F)
            obstacle: The obstacle mask (R8)
            cell: A cell's sides in widths, x, y and z
            sweeps: Red-black sweeps, each a red and a black half
            omega: The over-relaxation, 1 (Gauss–Seidel) to under 2
            has_obstacles: Whether obstacle logic is active
            closed: Per axis, whether the domain's walls are closed
        """
        if not self.allocated or not self.shader_program:
            return
        alpha = (1.0 / (cell[0] * cell[0]), 1.0 / (cell[1] * cell[1]), 1.0 / (cell[2] * cell[2]))
        glUseProgram(self.shader_program)
        self.bind_texture_3d(0, divergence, "uDivergence")
        self.bind_texture_3d(1, obstacle, "uObstacle")
        self.bind_image_3d_readwrite(0, pressure, GL_R32F)
        glUniform3f(self.get_uniform_loc("uAlpha"), *alpha)
        glUniform1f(self.get_uniform_loc("uBeta"), 1.0 / (2.0 * sum(alpha)))
        glUniform1f(self.get_uniform_loc("uOmega"), omega)
        glUniform3i(self.get_uniform_loc("uSize"), pressure.width, pressure.height, pressure.depth)
        glUniform1i(self.get_uniform_loc("uHasObstacles"), int(has_obstacles))
        glUniform3i(self.get_uniform_loc("uClosed"), *(int(c) for c in closed))
        color_loc = self.get_uniform_loc("uColor")
        half_width = (pressure.width + 1) // 2
        for _ in range(sweeps):
            for color in (0, 1):
                glUniform1i(color_loc, color)
                self.dispatch(half_width, pressure.height, pressure.depth)     # its barrier orders the halves
