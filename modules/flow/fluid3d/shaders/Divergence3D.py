"""Divergence3D - Compute divergence of 3D velocity field."""
from __future__ import annotations

from OpenGL.GL import *  # type: ignore

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
            grid_scale: float, aspect: float, depth_scale: float,
            has_obstacles: bool = True) -> None:
        """Compute divergence of 3D velocity field.

        Args:
            velocity: 3D velocity field (RGBA16F)
            obstacle: 3D obstacle mask (R8)
            divergence_out: Output divergence volume (R16F)
            grid_scale: Grid scaling factor
            aspect: Width/height ratio
            depth_scale: Z grid spacing relative to XY
            has_obstacles: Whether obstacle logic is active
        """
        if not self.allocated or not self.shader_program:
            return

        glUseProgram(self.shader_program)

        # Bind inputs as samplers
        self.bind_texture_3d(0, velocity, "uVelocity")
        self.bind_texture_3d(1, obstacle, "uObstacle")

        # Bind output as 3D image
        self.bind_image_3d_write(0, divergence_out, GL_R16F)

        # Aspect-corrected grid scales
        dx = grid_scale
        dy = grid_scale * aspect
        dz = grid_scale * depth_scale
        glUniform3f(self.get_uniform_loc("uHalfRdxInv"), 0.5 / dx, 0.5 / dy, 0.5 / dz)
        glUniform3i(self.get_uniform_loc("uSize"),
                    divergence_out.width, divergence_out.height, divergence_out.depth)
        glUniform1i(self.get_uniform_loc("uHasObstacles"), int(has_obstacles))

        self.dispatch(divergence_out.width, divergence_out.height, divergence_out.depth)
