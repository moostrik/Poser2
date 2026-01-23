"""JacobiPressure shader - Iterative Poisson pressure solver."""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture


class JacobiPressure(Shader):
    """Jacobi iterative solver for Poisson pressure equation."""

    def __init__(self) -> None:
        super().__init__()

    def use(self, source: Texture, divergence: Texture, obstacle: Texture, obstacle_offset: Texture, grid_scale: float, aspect: float) -> None:
        """Apply one Jacobi iteration.

        Args:
            source: Previous pressure estimate (R32F)
            divergence: Velocity divergence (R32F)
            obstacle: Obstacle mask (R8/R32F)
            obstacle_offset: Neighbor obstacle info (RGBA8)
            grid_scale: Grid scaling factor (typically 1)
            aspect: Aspect ratio (width/height)
        """
        if not self.allocated or not self.shader_program:
            print("JacobiPressure shader not allocated or shader program missing.")
            return
        if not source.allocated or not divergence.allocated or not obstacle.allocated or not obstacle_offset.allocated:
            print("JacobiPressure shader: input texture(s) not allocated.")
            return

        # Compute Jacobi parameters
        # Anisotropic grid spacing
        dx = grid_scale
        dy = grid_scale * aspect

        # Laplacian weights
        alpha_x = 1.0 / (dx * dx)
        alpha_y = 1.0 / (dy * dy)
        beta = 1.0 / (2.0 * alpha_x + 2.0 * alpha_y)

        glUseProgram(self.shader_program)

        # Bind textures
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, source.tex_id)
        glUniform1i(self.get_uniform_loc("uSource"), 0)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, divergence.tex_id)
        glUniform1i(self.get_uniform_loc("uDivergence"), 1)

        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, obstacle.tex_id)
        glUniform1i(self.get_uniform_loc("uObstacle"), 2)

        glActiveTexture(GL_TEXTURE3)
        glBindTexture(GL_TEXTURE_2D, obstacle_offset.tex_id)
        glUniform1i(self.get_uniform_loc("uObstacleOffset"), 3)

        # Set uniforms
        glUniform2f(self.get_uniform_loc("uAlpha"), alpha_x, alpha_y)
        glUniform1f(self.get_uniform_loc("uBeta"), beta)

        draw_quad()

        # Cleanup
        glActiveTexture(GL_TEXTURE3)
        glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glUseProgram(0)
