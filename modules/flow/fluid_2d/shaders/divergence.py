"""Divergence shader - Compute divergence of velocity field."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE2, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUniform2f, glUseProgram
from modules.gl import Shader, StorageBuffer, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)


class Divergence(Shader):
    """Compute velocity field divergence."""

    def __init__(self) -> None:
        super().__init__()

    def use(self, velocity: Texture, obstacle: Texture, cell: tuple[float, float],
            has_obstacles: bool = True, expansion: Texture | None = None, rate: float = 0.0,
            mean: StorageBuffer | None = None, closed: bool = True, wall_keep: float = 1.0) -> None:
        """Compute divergence.

        Args:
            velocity: Velocity field (RG32F)
            obstacle: Obstacle mask (R8, CLAMP_TO_BORDER=1)
            cell: A cell's width and height, widths
            has_obstacles: When False, skip obstacle texture reads for performance
            expansion: The expansion (R16F), its rate × it taken off the divergence; None none
            rate: The expansion's rate, 1/s
            mean: The expansion's mean (ExpansionMean2D's, one float), taken off first (closed walls); None none
            closed: The walls closed, else open (walls_2d.glsl)
            wall_keep: The share of the velocity along a closed wall kept a step; 1 free-slip
        """
        if not self.allocated or not self.shader_program:
            logger.warning("Divergence shader not allocated or shader program missing.")
            return
        if not velocity.allocated or not obstacle.allocated:
            logger.warning("Divergence shader: input texture(s) not allocated.")
            return

        glUseProgram(self.shader_program)

        # Obstacle flag
        glUniform1i(self.get_uniform_loc("uHasObstacles"), int(has_obstacles))
        glUniform1i(self.get_uniform_loc("uClosed"), int(closed))
        glUniform1f(self.get_uniform_loc("uWallKeep"), wall_keep)

        # Bind textures
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, velocity.tex_id)
        glUniform1i(self.get_uniform_loc("uVelocity"), 0)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, obstacle.tex_id)
        glUniform1i(self.get_uniform_loc("uObstacle"), 1)

        # 1 / (2 × the cell), widths: a central difference over two cells
        half_rdx_x = 0.5 / cell[0]
        half_rdx_y = 0.5 / cell[1]
        glUniform2f(self.get_uniform_loc("uHalfRdxInv"), half_rdx_x, half_rdx_y)

        has_expansion: bool = expansion is not None and expansion.allocated and rate != 0.0
        glUniform1i(self.get_uniform_loc("uHasExpansion"), int(has_expansion))
        if has_expansion:
            assert expansion is not None
            glActiveTexture(GL_TEXTURE2)
            glBindTexture(GL_TEXTURE_2D, expansion.tex_id)
            glUniform1i(self.get_uniform_loc("uExpansion"), 2)
            glUniform1f(self.get_uniform_loc("uRate"), rate)
            glUniform1i(self.get_uniform_loc("uSubtractMean"), int(mean is not None))
            if mean is not None:
                mean.bind_base(1)

        draw_quad()
