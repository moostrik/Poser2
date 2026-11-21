"""JacobiDiffusion shader - Iterative diffusion solver for velocity viscosity."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUniform2f, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class JacobiDiffusion(Shader):
    """Jacobi iterative solver for diffusion (viscosity)."""

    def __init__(self) -> None:
        super().__init__()

    def use(self, source: Texture, obstacle: Texture,
            cell: tuple[float, float], viscosity_dt: float,
            has_obstacles: bool = True, closed: bool = True, wall_keep: float = 1.0) -> None:
        """Apply one Jacobi iteration for diffusion.

        Args:
            source: Previous iteration of field to diffuse (velocity RG32F)
            obstacle: Obstacle mask (R8, CLAMP_TO_BORDER=1)
            cell: A cell's width and height, widths
            viscosity_dt: Viscosity * delta_time (diffusion rate)
            has_obstacles: When False, skip obstacle texture reads for performance
            closed: The walls closed, else open (walls_2d.glsl)
            wall_keep: The share of the velocity along a closed wall kept a step; 1 free-slip
        """
        # Anisotropic grid spacing
        dx, dy = cell

        # Laplacian weights: 1/dx², 1/dy²
        alpha_x = 1.0 / (dx * dx)
        alpha_y = 1.0 / (dy * dy)

        # Central coefficient: 1/(ν·Δt)
        gamma = 1.0 / max(viscosity_dt, 1e-6)

        # Beta = 1 / (2*alpha_x + 2*alpha_y + gamma)
        beta = 1.0 / (2.0 * alpha_x + 2.0 * alpha_y + gamma)

        # Bind shader program
        glUseProgram(self.shader_program)

        # Obstacle flag
        glUniform1i(self.get_uniform_loc("uHasObstacles"), int(has_obstacles))
        glUniform1i(self.get_uniform_loc("uClosed"), int(closed))
        glUniform1f(self.get_uniform_loc("uWallKeep"), wall_keep)

        # Bind textures
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, source.tex_id)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, obstacle.tex_id)

        # Set uniforms
        glUniform2f(self.get_uniform_loc("uAlpha"), alpha_x, alpha_y)
        glUniform1f(self.get_uniform_loc("uGamma"), gamma)
        glUniform1f(self.get_uniform_loc("uBeta"), beta)

        # Draw fullscreen quad
        draw_quad()
