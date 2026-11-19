"""SorPressure2D - the 2D pressure by red-black Gauss–Seidel with over-relaxation, in place."""
from __future__ import annotations

from OpenGL.GL import GL_R32F, glUniform1f, glUniform1i, glUniform2f, glUniform2i, glUseProgram

from modules.gl import Texture, ComputeShader


class SorPressure2D(ComputeShader):
    """Red-black SOR on the 2D pressure: each sweep a red and a black half, each color from the other's newest values."""

    WORKGROUP_SIZE_X = 16
    WORKGROUP_SIZE_Y = 16
    WORKGROUP_SIZE_Z = 1

    def solve(self, pressure: Texture, divergence: Texture, obstacle: Texture, cell: tuple[float, float], sweeps: int,
              omega: float, has_obstacles: bool = True, closed: bool = True) -> None:
        """`sweeps` red-black sweeps on `pressure` in place, warm from its last values.

        Args:
            pressure: The pressure (R32F), read and written
            divergence: The velocity's divergence, the right-hand side
            obstacle: The obstacle mask (R8)
            cell: A cell's width and height, widths
            sweeps: Red-black sweeps, each a red and a black half
            omega: The over-relaxation, 1 (Gauss–Seidel) to under 2
            has_obstacles: Whether obstacle logic is active
            closed: The walls closed, else open
        """
        if not self.allocated or not self.shader_program:
            return
        alpha = (1.0 / (cell[0] * cell[0]), 1.0 / (cell[1] * cell[1]))
        glUseProgram(self.shader_program)
        self.bind_texture(0, divergence, "uDivergence")
        self.bind_texture(1, obstacle, "uObstacle")
        self.bind_image_readwrite(0, pressure, GL_R32F)
        glUniform2f(self.get_uniform_loc("uAlpha"), *alpha)
        glUniform1f(self.get_uniform_loc("uBeta"), 1.0 / (2.0 * sum(alpha)))
        glUniform1f(self.get_uniform_loc("uOmega"), omega)
        glUniform2i(self.get_uniform_loc("uSize"), pressure.width, pressure.height)
        glUniform1i(self.get_uniform_loc("uHasObstacles"), int(has_obstacles))
        glUniform1i(self.get_uniform_loc("uClosed"), int(closed))
        color_loc = self.get_uniform_loc("uColor")
        half_width = (pressure.width + 1) // 2
        for _ in range(sweeps):
            for color in (0, 1):
                glUniform1i(color_loc, color)
                self.dispatch(half_width, pressure.height)     # its barrier orders the halves
