"""Gradient shader - Subtract pressure gradient from velocity."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE2, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1i, glUniform2f, glUseProgram
from modules.gl import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)


class Gradient(Shader):
    """Subtract pressure gradient from velocity (projection step)."""

    def __init__(self) -> None:
        super().__init__()

    def use(self, velocity: Texture, pressure: Texture, obstacle: Texture, cell: tuple[float, float],
            has_obstacles: bool = True, closed: bool = True) -> None:
        """Apply pressure gradient subtraction.

        Args:
            velocity: Current velocity field (RG32F)
            pressure: Pressure field (R32F)
            obstacle: Obstacle mask (R8, CLAMP_TO_BORDER=1)
            cell: A cell's width and height, widths
            has_obstacles: When False, skip obstacle texture reads for performance
            closed: The walls closed (the pressure beyond them the cell's), else open (its negative beyond them)
        """
        if not self.allocated or not self.shader_program:
            logger.warning("Gradient shader not allocated or shader program missing.")
            return
        if not velocity.allocated or not pressure.allocated or not obstacle.allocated:
            logger.warning("Gradient shader: input texture(s) not allocated.")
            return

        glUseProgram(self.shader_program)

        # Obstacle flag
        glUniform1i(self.get_uniform_loc("uHasObstacles"), int(has_obstacles))
        glUniform1i(self.get_uniform_loc("uClosed"), int(closed))

        # Bind textures
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, velocity.tex_id)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, pressure.tex_id)

        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, obstacle.tex_id)

        # 1 / (2 × the cell), widths: a central difference over two cells
        half_rdx_x = 0.5 / cell[0]
        half_rdx_y = 0.5 / cell[1]
        glUniform2f(self.get_uniform_loc("uHalfRdxInv"), half_rdx_x, half_rdx_y)

        draw_quad()
