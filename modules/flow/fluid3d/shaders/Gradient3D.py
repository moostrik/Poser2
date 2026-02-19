"""Gradient3D - Subtract 3D pressure gradient from velocity."""
from __future__ import annotations

from OpenGL.GL import *  # type: ignore

from modules.gl.Texture3D import Texture3D
from modules.gl.ComputeShader import ComputeShader


class Gradient3D(ComputeShader):
    """Subtract 3D pressure gradient from velocity (projection step)."""

    WORKGROUP_SIZE_X = 16
    WORKGROUP_SIZE_Y = 16
    WORKGROUP_SIZE_Z = 1

    def __init__(self) -> None:
        super().__init__()

    def use(self, velocity: Texture3D, pressure: Texture3D, obstacle: Texture3D,
            result_out: Texture3D,
            grid_scale: float, aspect: float, depth_scale: float) -> None:
        """Subtract pressure gradient: v_new = v_old - grad(p).

        Args:
            velocity: Current 3D velocity field (RGBA16F)
            pressure: 3D pressure field (R16F)
            obstacle: 3D obstacle mask (R8)
            result_out: Output corrected velocity (RGBA16F)
            grid_scale: Grid scaling factor
            aspect: Width/height ratio
            depth_scale: Z grid spacing relative to XY
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
        dz = grid_scale * depth_scale
        glUniform3f(self.get_uniform_loc("uHalfRdxInv"), 0.5 / dx, 0.5 / dy, 0.5 / dz)
        glUniform3i(self.get_uniform_loc("uSize"),
                    result_out.width, result_out.height, result_out.depth)

        self.dispatch(result_out.width, result_out.height, result_out.depth)
