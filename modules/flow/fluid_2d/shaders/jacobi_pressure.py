"""JacobiPressure shader - Iterative Poisson pressure solver."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE2, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUniform2f, glUseProgram
from modules.gl import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)


class JacobiPressure(Shader):
    """Jacobi iterative solver for Poisson pressure equation."""

    def __init__(self) -> None:
        super().__init__()

    def use(self, source: Texture, divergence: Texture, obstacle: Texture, cell: tuple[float, float],
            has_obstacles: bool = True, closed: bool = True, screening: float = 0.0) -> None:
        """Apply one Jacobi iteration.

        Args:
            source: Previous pressure estimate (R32F)
            divergence: Velocity divergence (R32F)
            obstacle: Obstacle mask (R8, CLAMP_TO_BORDER=1)
            cell: A cell's width and height, widths
            has_obstacles: When False, skip obstacle texture reads for performance
            closed: The walls closed (the pressure beyond them the cell's), else open (its negative beyond them)
        """
        if not self.allocated or not self.shader_program:
            logger.warning("JacobiPressure shader not allocated or shader program missing.")
            return
        if not source.allocated or not divergence.allocated or not obstacle.allocated:
            logger.warning("JacobiPressure shader: input texture(s) not allocated.")
            return

        # The Laplacian over the cell, widths
        dx, dy = cell

        # Laplacian weights
        alpha_x = 1.0 / (dx * dx)
        alpha_y = 1.0 / (dy * dy)
        beta = 1.0 / (2.0 * alpha_x + 2.0 * alpha_y + screening)

        glUseProgram(self.shader_program)

        # Obstacle flag
        glUniform1i(self.get_uniform_loc("uHasObstacles"), int(has_obstacles))
        glUniform1i(self.get_uniform_loc("uClosed"), int(closed))

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

        # Set uniforms
        glUniform2f(self.get_uniform_loc("uAlpha"), alpha_x, alpha_y)
        glUniform1f(self.get_uniform_loc("uBeta"), beta)

        draw_quad()
