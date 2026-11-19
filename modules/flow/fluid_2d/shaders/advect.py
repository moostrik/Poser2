"""Advect shader - Semi-Lagrangian advection with the fade, the evaporation and the guard.

Advection formula:  ΔUV = dt × advection_scale × velocity × (1, aspect), the velocity in widths/s: x is uv of the
width, y of the height, where one width is `aspect` (the field's width over its height) of the height. At scale 1 a
velocity of 1 crosses the width in 1 second, and the same distance on y.
"""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE2, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUniform2f, glUseProgram
from modules.gl import Shader, draw_quad, Texture

from ...finite_guard import GuardSlot

import logging
logger = logging.getLogger(__name__)


class Advect(Shader):
    """Semi-Lagrangian advection shader with the fade, the evaporation, the guard and aspect correction."""

    def use(self, source: Texture, velocity: Texture, obstacle: Texture,
            aspect: float, timestep: float, dissipation: float,
            has_obstacles: bool = True, evaporation: float = 0.0, evaporation_knee: float = 0.1, measure: int = 0,
            guard: GuardSlot | None = None, closed: bool = True, wall_keep: float = 1.0) -> None:
        """Apply advection.

        Args:
            source: Field to advect (density, velocity, temperature)
            velocity: Velocity field (RG16F), widths/s
            obstacle: Obstacle mask (R8/R32F)
            aspect: The field's width over its height: one width is this share of the height
            timestep: dt × advection_scale
            dissipation: The fade's multiplier this step; 1 = none
            has_obstacles: When False, skip obstacle texture reads for performance
            evaporation: dt / evaporation_time: a full value's share lost this step well above the knee, 0 off
            evaporation_knee: The size below which the loss turns from a fixed amount to a share
            measure: How the size is read: 0 a premultiplied density's alpha, 1 a vector's length, 2 |r|
            guard: The field's slot in its stage's FiniteGuard: a non-finite value 0 and counted; None no guard
            closed: The walls closed, else open (walls_2d.glsl)
            wall_keep: The velocity's share along a closed wall kept this step, its cells against the wall braked;
                1 for every field but the velocity
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
        glUniform1i(self.get_uniform_loc("uClosed"), int(closed))
        glUniform1f(self.get_uniform_loc("uWallKeep"), wall_keep)

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
        glUniform2f(self.get_uniform_loc("uRdx"), 1.0, aspect if aspect > 0.0 else 1.0)
        glUniform1f(self.get_uniform_loc("uDissipation"), dissipation)
        glUniform1f(self.get_uniform_loc("uEvaporation"), evaporation)
        glUniform1f(self.get_uniform_loc("uEvaporationKnee"), evaporation_knee)
        glUniform1i(self.get_uniform_loc("uMeasure"), measure)
        if guard is not None:
            guard.use(self.get_uniform_loc("uGuardSlot"))
        else:
            glUniform1i(self.get_uniform_loc("uGuardSlot"), -1)

        draw_quad()
