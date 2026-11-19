"""Advect shader - Semi-Lagrangian advection with dissipation.

Advection formula:  ΔUV = dt × speed × velocity × (1, aspect), the velocity in widths/s: x is uv of the width, y of the
height, where one width is `aspect` (the field's width over its height) of the height. At speed 1 a
velocity of 1 crosses the width in 1 second, and the same distance on y.
"""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE2, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUniform2f, glUseProgram
from modules.gl import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)


class Advect(Shader):
    """Semi-Lagrangian advection shader with dissipation and aspect correction."""

    def __init__(self) -> None:
        super().__init__()

    def use(self, source: Texture, velocity: Texture, obstacle: Texture,
            aspect: float, timestep: float, dissipation: float,
            has_obstacles: bool = True) -> None:
        """Apply advection.

        Args:
            source: Field to advect (density, velocity, temperature, pressure)
            velocity: Velocity field (RG32F)
            obstacle: Obstacle mask (R8/R32F)
            aspect: The field's width over its height: one width is this share of the height
            timestep: dt × speed — advection distance per frame in UV space
            dissipation: Exponential decay multiplier (0.99 = 1% loss per frame)
            has_obstacles: When False, skip obstacle texture reads for performance
        """
        if not self.allocated or not self.shader_program:
            logger.warning("Advect shader not allocated or shader program missing.")
            return
        if not source.allocated or not velocity.allocated or not obstacle.allocated:
            logger.warning("Advect shader: input texture(s) not allocated.")
            return

        glUseProgram(self.shader_program)

        # Obstacle flag
        glUniform1i(self.get_uniform_loc("uHasObstacles"), int(has_obstacles))

        # Bind textures
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, velocity.tex_id)
        glUniform1i(self.get_uniform_loc("uVelocity"), 0)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, source.tex_id)
        glUniform1i(self.get_uniform_loc("uSource"), 1)

        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, obstacle.tex_id)
        glUniform1i(self.get_uniform_loc("uObstacle"), 2)

        # Set uniforms
        glUniform1f(self.get_uniform_loc("uTimestep"), timestep)
        # uv per width: x the width itself, y the share of the height one width is
        rdx_x = 1.0
        rdx_y = aspect if aspect > 0.0 else 1.0
        glUniform2f(self.get_uniform_loc("uRdx"), rdx_x, rdx_y)
        glUniform1f(self.get_uniform_loc("uDissipation"), dissipation)

        draw_quad()
