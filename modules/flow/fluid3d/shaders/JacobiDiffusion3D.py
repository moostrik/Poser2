"""JacobiDiffusion3D - 3D Jacobi diffusion solver for velocity viscosity."""
from __future__ import annotations

import math

from OpenGL.GL import *  # type: ignore

from modules.gl.Texture3D import Texture3D
from modules.gl.ComputeShader import ComputeShader


class JacobiDiffusion3D(ComputeShader):
    """3D Jacobi diffusion solver for velocity viscosity.

    Multi-iteration with shared memory tiling in XY plane.
    """

    WORKGROUP_SIZE_X = 16
    WORKGROUP_SIZE_Y = 16
    WORKGROUP_SIZE_Z = 1

    DEFAULT_ITERATIONS_PER_DISPATCH = 5

    def __init__(self) -> None:
        super().__init__()

    def use(self, velocity_in: Texture3D, velocity_out: Texture3D,
            obstacle: Texture3D,
            grid_scale: float, aspect: float, depth_scale: float,
            viscosity_dt: float,
            iterations: int = DEFAULT_ITERATIONS_PER_DISPATCH) -> None:
        """Run multiple 3D Jacobi diffusion iterations.

        Args:
            velocity_in: Input velocity volume (RGBA16F)
            velocity_out: Output velocity volume (RGBA16F)
            obstacle: 3D obstacle mask (R8)
            grid_scale: Grid scaling factor
            aspect: Width/height ratio
            depth_scale: Z grid spacing relative to XY
            viscosity_dt: Viscosity * delta_time
            iterations: Iterations per dispatch
        """
        if not self.allocated or not self.shader_program:
            return

        dx = grid_scale
        dy = grid_scale * aspect
        dz = grid_scale * depth_scale

        alpha_x = 1.0 / (dx * dx)
        alpha_y = 1.0 / (dy * dy)
        alpha_z = 1.0 / (dz * dz)
        gamma = 1.0 / max(viscosity_dt, 1e-6)
        beta = 1.0 / (2.0 * alpha_x + 2.0 * alpha_y + 2.0 * alpha_z + gamma)

        glUseProgram(self.shader_program)

        self.bind_texture_3d(0, obstacle, "uObstacle")

        self.bind_image_3d(0, velocity_in, GL_READ_ONLY, GL_RGBA16F)
        self.bind_image_3d(1, velocity_out, GL_WRITE_ONLY, GL_RGBA16F)

        glUniform3f(self.get_uniform_loc("uAlpha"), alpha_x, alpha_y, alpha_z)
        glUniform1f(self.get_uniform_loc("uGamma"), gamma)
        glUniform1f(self.get_uniform_loc("uBeta"), beta)
        glUniform1i(self.get_uniform_loc("uIterations"), iterations)
        glUniform3i(self.get_uniform_loc("uSize"),
                    velocity_in.width, velocity_in.height, velocity_in.depth)

        self.dispatch(velocity_in.width, velocity_in.height, velocity_in.depth)

    def solve(self, velocity_a: Texture3D, velocity_b: Texture3D,
              obstacle: Texture3D,
              grid_scale: float, aspect: float, depth_scale: float,
              viscosity_dt: float,
              total_iterations: int = 20,
              iterations_per_dispatch: int = DEFAULT_ITERATIONS_PER_DISPATCH) -> Texture3D:
        """Full diffusion solve with automatic ping-pong.

        Returns:
            The Texture3D containing the final velocity result
        """
        num_dispatches = math.ceil(total_iterations / iterations_per_dispatch)

        src = velocity_a
        dst = velocity_b

        for i in range(num_dispatches):
            iters = min(iterations_per_dispatch,
                        total_iterations - i * iterations_per_dispatch)
            self.use(src, dst, obstacle,
                     grid_scale, aspect, depth_scale, viscosity_dt, iters)
            src, dst = dst, src

        return velocity_a if (num_dispatches % 2 == 0) else velocity_b
