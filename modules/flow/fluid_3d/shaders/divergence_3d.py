"""Divergence3D - Compute divergence of 3D velocity field."""
from __future__ import annotations

from OpenGL.GL import GL_R16F, glUniform1i, glUniform3f, glUniform3i, glUseProgram

from modules.gl import Texture3D, ComputeShader


class Divergence3D(ComputeShader):
    """Compute 3D velocity divergence: div(v) = du/dx + dv/dy + dw/dz."""

    WORKGROUP_SIZE_X = 16
    WORKGROUP_SIZE_Y = 16
    WORKGROUP_SIZE_Z = 1

    def __init__(self) -> None:
        super().__init__()

    def use(self, velocity: Texture3D, obstacle: Texture3D,
            divergence_out: Texture3D,
            cell: float,
            has_obstacles: bool = True, closed: tuple[bool, bool, bool] = (True, True, True)) -> None:
        """Compute divergence of 3D velocity field.

        Args:
            velocity: 3D velocity field (RGBA16F)
            obstacle: 3D obstacle mask (R8)
            divergence_out: Output divergence volume (R16F)
            cell: The cubic cell's size in widths
            has_obstacles: Whether obstacle logic is active
            closed: Per axis, whether the domain's walls are closed (solid) or open
        """
        if not self.allocated or not self.shader_program:
            return

        glUseProgram(self.shader_program)

        # Bind inputs as samplers
        self.bind_texture_3d(0, velocity, "uVelocity")
        self.bind_texture_3d(1, obstacle, "uObstacle")

        # Bind output as 3D image
        self.bind_image_3d_write(0, divergence_out, GL_R16F)

        # Aspect-corrected grid scales; the cells are cubic, so dz = dx
        dx = cell                         # cubic cells, in widths
        dy = dz = cell
        glUniform3f(self.get_uniform_loc("uHalfRdxInv"), 0.5 / dx, 0.5 / dy, 0.5 / dz)
        glUniform3i(self.get_uniform_loc("uSize"),
                    divergence_out.width, divergence_out.height, divergence_out.depth)
        glUniform1i(self.get_uniform_loc("uHasObstacles"), int(has_obstacles))
        glUniform3i(self.get_uniform_loc("uClosed"), *(int(c) for c in closed))

        self.dispatch(divergence_out.width, divergence_out.height, divergence_out.depth)
