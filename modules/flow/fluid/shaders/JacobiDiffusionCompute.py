"""JacobiDiffusionCompute - Compute shader diffusion solver.

Multi-iteration Jacobi solver for velocity viscosity using shared memory tiling.
Performs multiple iterations per dispatch, reducing the number of texture swaps
from 20 to ~4-5 dispatches.

Uses 16x16 workgroups with 1-pixel halo (18x18 shared memory) for fast neighbor access.
"""
from __future__ import annotations

from OpenGL.GL import *  # type: ignore

from modules.gl import Texture
from modules.gl.ComputeShader import ComputeShader


class JacobiDiffusionCompute(ComputeShader):
    """Compute shader Jacobi solver for diffusion with shared memory tiling.

    Performs multiple Jacobi iterations per dispatch using shared memory,
    dramatically reducing global memory bandwidth and FBO swap overhead.
    """

    # Workgroup configuration (must match .comp file)
    WORKGROUP_SIZE_X = 16
    WORKGROUP_SIZE_Y = 16
    WORKGROUP_SIZE_Z = 1

    # Iterations per dispatch
    DEFAULT_ITERATIONS_PER_DISPATCH = 5

    def __init__(self) -> None:
        super().__init__()

    def use(
        self,
        velocity_in: Texture,
        velocity_out: Texture,
        obstacle: Texture,
        obstacle_offset: Texture,
        grid_scale: float,
        aspect: float,
        viscosity_dt: float,
        iterations: int = DEFAULT_ITERATIONS_PER_DISPATCH
    ) -> None:
        """Run multiple Jacobi diffusion iterations in a single dispatch.

        Args:
            velocity_in: Input velocity field (RG32F) - read via image
            velocity_out: Output velocity field (RG32F) - write via image
            obstacle: Obstacle mask (R8/R32F) - read via sampler
            obstacle_offset: Neighbor obstacle info (RGBA8) - read via sampler
            grid_scale: Grid scaling factor
            aspect: Aspect ratio (width/height)
            viscosity_dt: Viscosity * delta_time (diffusion rate)
            iterations: Number of Jacobi iterations to perform in this dispatch
        """
        if not self.allocated or not self.shader_program:
            print("JacobiDiffusionCompute shader not allocated.")
            return

        # Validate inputs
        if not all(t.allocated for t in [velocity_in, velocity_out, obstacle, obstacle_offset]):
            print("JacobiDiffusionCompute: input texture(s) not allocated.")
            return

        # Compute Jacobi parameters (same as fragment shader version)
        dx = grid_scale
        dy = grid_scale * aspect

        # Laplacian weights
        alpha_x = 1.0 / (dx * dx)
        alpha_y = 1.0 / (dy * dy)

        # Central coefficient
        gamma = 1.0 / max(viscosity_dt, 1e-6)

        # Beta = 1 / (2*alpha_x + 2*alpha_y + gamma)
        beta = 1.0 / (2.0 * alpha_x + 2.0 * alpha_y + gamma)

        glUseProgram(self.shader_program)

        # Bind textures via samplers
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, obstacle.tex_id)
        glUniform1i(self.get_uniform_loc("uObstacle"), 0)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, obstacle_offset.tex_id)
        glUniform1i(self.get_uniform_loc("uObstacleOffset"), 1)

        # Bind velocity textures as images
        self.bind_image_read(0, velocity_in, GL_RG16F)
        self.bind_image_write(1, velocity_out, GL_RG16F)

        # Set uniforms
        glUniform2f(self.get_uniform_loc("uAlpha"), alpha_x, alpha_y)
        glUniform1f(self.get_uniform_loc("uGamma"), gamma)
        glUniform1f(self.get_uniform_loc("uBeta"), beta)
        glUniform1i(self.get_uniform_loc("uIterations"), iterations)
        glUniform2i(self.get_uniform_loc("uSize"), velocity_in.width, velocity_in.height)

        # Dispatch compute shader
        self.dispatch(velocity_in.width, velocity_in.height)

    def solve(
        self,
        velocity_a: Texture,
        velocity_b: Texture,
        obstacle: Texture,
        obstacle_offset: Texture,
        grid_scale: float,
        aspect: float,
        viscosity_dt: float,
        total_iterations: int = 20,
        iterations_per_dispatch: int = DEFAULT_ITERATIONS_PER_DISPATCH
    ) -> Texture:
        """Run full diffusion solve with automatic ping-pong.

        Args:
            velocity_a: First velocity buffer (RG32F)
            velocity_b: Second velocity buffer (RG32F) - used for ping-pong
            obstacle: Obstacle mask
            obstacle_offset: Neighbor obstacle info
            grid_scale: Grid scaling factor
            aspect: Aspect ratio
            viscosity_dt: Viscosity * delta_time
            total_iterations: Total Jacobi iterations to perform
            iterations_per_dispatch: Iterations per compute dispatch

        Returns:
            The texture containing the final velocity result
        """
        import math
        num_dispatches = math.ceil(total_iterations / iterations_per_dispatch)

        # Ping-pong between buffers
        src = velocity_a
        dst = velocity_b

        for i in range(num_dispatches):
            # Last dispatch may have fewer iterations
            iters = min(iterations_per_dispatch, total_iterations - i * iterations_per_dispatch)

            self.use(
                src, dst,
                obstacle, obstacle_offset,
                grid_scale, aspect,
                viscosity_dt,
                iters
            )

            # Swap buffers for next iteration
            src, dst = dst, src

        # Return the buffer that has the final result
        return velocity_a if (num_dispatches % 2 == 0) else velocity_b
