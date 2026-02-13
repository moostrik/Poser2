"""JacobiPressureCompute - Compute shader Jacobi pressure solver.

Multi-iteration Jacobi solver using shared memory tiling for improved performance.
Performs multiple iterations per dispatch, reducing the number of texture swaps
from 40 to ~5-10 dispatches.

Uses 16x16 workgroups with 1-pixel halo (18x18 shared memory) for fast neighbor access.
"""
from __future__ import annotations

from OpenGL.GL import *  # type: ignore

from modules.gl import Texture
from modules.gl.ComputeShader import ComputeShader


class JacobiPressureCompute(ComputeShader):
    """Compute shader Jacobi solver with shared memory tiling.

    Performs multiple Jacobi iterations per dispatch using shared memory,
    dramatically reducing global memory bandwidth and FBO swap overhead.
    """

    # Workgroup configuration (must match .comp file)
    WORKGROUP_SIZE_X = 16
    WORKGROUP_SIZE_Y = 16
    WORKGROUP_SIZE_Z = 1

    # Iterations per dispatch (balance between shared memory efficiency and convergence)
    # More iterations = fewer dispatches, but halo becomes stale
    # 5-8 is typically optimal for 16x16 tiles
    DEFAULT_ITERATIONS_PER_DISPATCH = 5

    def __init__(self) -> None:
        super().__init__()

    def use(
        self,
        pressure_in: Texture,
        pressure_out: Texture,
        divergence: Texture,
        obstacle: Texture,
        obstacle_offset: Texture,
        grid_scale: float,
        aspect: float,
        iterations: int = DEFAULT_ITERATIONS_PER_DISPATCH
    ) -> None:
        """Run multiple Jacobi iterations in a single dispatch.

        Args:
            pressure_in: Input pressure field (R32F) - read via image
            pressure_out: Output pressure field (R32F) - write via image
            divergence: Velocity divergence (R32F) - read via sampler
            obstacle: Obstacle mask (R8/R32F) - read via sampler
            obstacle_offset: Neighbor obstacle info (RGBA8) - read via sampler
            grid_scale: Grid scaling factor
            aspect: Aspect ratio (width/height)
            iterations: Number of Jacobi iterations to perform in this dispatch

        Note:
            For 40 total iterations with iterations=5, call this 8 times,
            swapping pressure_in/pressure_out each call.
        """
        if not self.allocated or not self.shader_program:
            print("JacobiPressureCompute shader not allocated.")
            return

        # Validate inputs
        if not all(t.allocated for t in [pressure_in, pressure_out, divergence, obstacle, obstacle_offset]):
            print("JacobiPressureCompute: input texture(s) not allocated.")
            return

        # Compute Jacobi parameters (same as fragment shader version)
        dx = grid_scale
        dy = grid_scale * aspect

        # Laplacian weights
        alpha_x = 1.0 / (dx * dx)
        alpha_y = 1.0 / (dy * dy)
        beta = 1.0 / (2.0 * alpha_x + 2.0 * alpha_y)

        glUseProgram(self.shader_program)

        # Bind textures via samplers (for filtered access to divergence, obstacles)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, divergence.tex_id)
        glUniform1i(self.get_uniform_loc("uDivergence"), 0)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, obstacle.tex_id)
        glUniform1i(self.get_uniform_loc("uObstacle"), 1)

        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, obstacle_offset.tex_id)
        glUniform1i(self.get_uniform_loc("uObstacleOffset"), 2)

        # Bind pressure textures as images for imageLoad/imageStore
        self.bind_image_read(0, pressure_in, GL_R16F)
        self.bind_image_write(1, pressure_out, GL_R16F)

        # Set uniforms
        glUniform2f(self.get_uniform_loc("uAlpha"), alpha_x, alpha_y)
        glUniform1f(self.get_uniform_loc("uBeta"), beta)
        glUniform1i(self.get_uniform_loc("uIterations"), iterations)
        glUniform2i(self.get_uniform_loc("uSize"), pressure_in.width, pressure_in.height)

        # Dispatch compute shader
        self.dispatch(pressure_in.width, pressure_in.height)

    def solve(
        self,
        pressure_a: Texture,
        pressure_b: Texture,
        divergence: Texture,
        obstacle: Texture,
        obstacle_offset: Texture,
        grid_scale: float,
        aspect: float,
        total_iterations: int = 40,
        iterations_per_dispatch: int = DEFAULT_ITERATIONS_PER_DISPATCH
    ) -> Texture:
        """Run full pressure solve with automatic ping-pong.

        Convenience method that handles buffer swapping automatically.

        Args:
            pressure_a: First pressure buffer (R32F)
            pressure_b: Second pressure buffer (R32F) - used for ping-pong
            divergence: Velocity divergence (R32F)
            obstacle: Obstacle mask
            obstacle_offset: Neighbor obstacle info
            grid_scale: Grid scaling factor
            aspect: Aspect ratio
            total_iterations: Total Jacobi iterations to perform
            iterations_per_dispatch: Iterations per compute dispatch

        Returns:
            The texture containing the final pressure result (either pressure_a or pressure_b)
        """
        import math
        num_dispatches = math.ceil(total_iterations / iterations_per_dispatch)

        # Ping-pong between buffers
        src = pressure_a
        dst = pressure_b

        for i in range(num_dispatches):
            # Last dispatch may have fewer iterations
            iters = min(iterations_per_dispatch, total_iterations - i * iterations_per_dispatch)

            self.use(
                src, dst,
                divergence, obstacle, obstacle_offset,
                grid_scale, aspect,
                iters
            )

            # Swap buffers for next iteration
            src, dst = dst, src

        # Return the buffer that has the final result
        # After even number of dispatches: result in pressure_a
        # After odd number of dispatches: result in pressure_b
        return pressure_a if (num_dispatches % 2 == 0) else pressure_b
