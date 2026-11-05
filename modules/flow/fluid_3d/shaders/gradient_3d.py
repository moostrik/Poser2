"""Gradient3D - Subtract 3D pressure gradient from velocity."""
from __future__ import annotations

from OpenGL.GL import GL_RGBA16F, glUniform1i, glUniform3f, glUniform3i, glUseProgram

from modules.gl import Texture3D, ComputeShader


class Gradient3D(ComputeShader):
    """Subtract 3D pressure gradient from velocity (projection step)."""

    WORKGROUP_SIZE_X = 16
    WORKGROUP_SIZE_Y = 16
    WORKGROUP_SIZE_Z = 1

    def __init__(self) -> None:
        super().__init__()

    def use(self, velocity: Texture3D, pressure: Texture3D, obstacle: Texture3D,
            result_out: Texture3D,
            grid_scale: float, aspect: float, z_spacing: float,
            has_obstacles: bool = True) -> None:
        """Subtract pressure gradient: v_new = v_old - grad(p).

        Args:
            velocity: Current 3D velocity field (RGBA16F)
            pressure: 3D pressure field (R16F)
            obstacle: 3D obstacle mask (R8)
            result_out: Output corrected velocity (RGBA16F)
            grid_scale: Grid scaling factor
            aspect: Width/height ratio
            z_spacing: Sim-grid Z cell size in sim texels (dz/dx)
            has_obstacles: Whether obstacle logic is active
        """
        if not self.allocated or not self.shader_program:
            return

        glUseProgram(self.shader_program)

        self.bind_texture_3d(0, velocity, "uVelocity")
        self.bind_texture_3d(1, pressure, "uPressure")
        self.bind_texture_3d(2, obstacle, "uObstacle")

        self.bind_image_3d_write(0, result_out, GL_RGBA16F)

        dx = grid_scale
        dy = grid_scale * aspect
        dz = grid_scale * z_spacing
        glUniform3f(self.get_uniform_loc("uHalfRdxInv"), 0.5 / dx, 0.5 / dy, 0.5 / dz)
        glUniform3i(self.get_uniform_loc("uSize"),
                    result_out.width, result_out.height, result_out.depth)
        glUniform1i(self.get_uniform_loc("uHasObstacles"), int(has_obstacles))

        self.dispatch(result_out.width, result_out.height, result_out.depth)
