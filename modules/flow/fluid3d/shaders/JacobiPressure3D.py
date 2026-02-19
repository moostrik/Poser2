"""JacobiPressure3D - 3D Jacobi pressure solver with shared memory tiling."""
from __future__ import annotations

import math

from OpenGL.GL import *  # type: ignore

from modules.gl.Texture3D import Texture3D
from modules.gl.ComputeShader import ComputeShader


class JacobiPressure3D(ComputeShader):
    """3D Jacobi pressure solver with shared memory tiling in XY.

    Multi-iteration per dispatch. Z neighbors loaded from global memory.
    """

    WORKGROUP_SIZE_X = 16
    WORKGROUP_SIZE_Y = 16
    WORKGROUP_SIZE_Z = 1

    DEFAULT_ITERATIONS_PER_DISPATCH = 5

    def __init__(self) -> None:
        super().__init__()

    def use(self, pressure_in: Texture3D, pressure_out: Texture3D,
            divergence: Texture3D, obstacle: Texture3D,
            grid_scale: float, aspect: float, depth_scale: float,
            iterations: int = DEFAULT_ITERATIONS_PER_DISPATCH) -> None:
        """Run multiple 3D Jacobi pressure iterations.

        Args:
            pressure_in: Input pressure volume (R16F) — read via image
            pressure_out: Output pressure volume (R16F) — write via image
            divergence: Velocity divergence volume (R16F) — read via sampler
            obstacle: 3D obstacle mask (R8) — read via sampler
            grid_scale: Grid scaling factor
            aspect: Width/height ratio
            depth_scale: Z grid spacing relative to XY
            iterations: Number of iterations per dispatch
        """
        if not self.allocated or not self.shader_program:
            return

        dx = grid_scale
        dy = grid_scale * aspect
        dz = grid_scale * depth_scale

        alpha_x = 1.0 / (dx * dx)
        alpha_y = 1.0 / (dy * dy)
        alpha_z = 1.0 / (dz * dz)
        beta = 1.0 / (2.0 * alpha_x + 2.0 * alpha_y + 2.0 * alpha_z)

        glUseProgram(self.shader_program)

        # Samplers
        self.bind_texture_3d(0, divergence, "uDivergence")
        self.bind_texture_3d(1, obstacle, "uObstacle")

        # Images
        self.bind_image_3d(0, pressure_in, GL_READ_ONLY, GL_R16F)
        self.bind_image_3d(1, pressure_out, GL_WRITE_ONLY, GL_R16F)

        # Uniforms
        glUniform3f(self.get_uniform_loc("uAlpha"), alpha_x, alpha_y, alpha_z)
        glUniform1f(self.get_uniform_loc("uBeta"), beta)
        glUniform1i(self.get_uniform_loc("uIterations"), iterations)
        glUniform3i(self.get_uniform_loc("uSize"),
                    pressure_in.width, pressure_in.height, pressure_in.depth)

        self.dispatch(pressure_in.width, pressure_in.height, pressure_in.depth)

    def solve(self, pressure_a: Texture3D, pressure_b: Texture3D,
              divergence: Texture3D, obstacle: Texture3D,
              grid_scale: float, aspect: float, depth_scale: float,
              total_iterations: int = 40,
              iterations_per_dispatch: int = DEFAULT_ITERATIONS_PER_DISPATCH) -> Texture3D:
        """Full pressure solve with automatic ping-pong.

        Args:
            pressure_a, pressure_b: Ping-pong pressure buffers
            divergence: Velocity divergence volume
            obstacle: 3D obstacle mask
            grid_scale, aspect, depth_scale: Grid parameters
            total_iterations: Total Jacobi iterations
            iterations_per_dispatch: Iterations per compute dispatch

        Returns:
            The Texture3D containing the final pressure result
        """
        num_dispatches = math.ceil(total_iterations / iterations_per_dispatch)

        src = pressure_a
        dst = pressure_b

        for i in range(num_dispatches):
            iters = min(iterations_per_dispatch,
                        total_iterations - i * iterations_per_dispatch)
            self.use(src, dst, divergence, obstacle,
                     grid_scale, aspect, depth_scale, iters)
            src, dst = dst, src

        return pressure_a if (num_dispatches % 2 == 0) else pressure_b
