"""MacCormack2D - the MacCormack correction of a 2D semi-Lagrangian advection, limited (Selle et al. 2008)."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE2, GL_TEXTURE3, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUniform2f, glUseProgram
from modules.gl import Shader, draw_quad, Texture

from ...finite_guard import GuardSlot


class MacCormack2D(Shader):
    """Correct a semi-Lagrangian predictor by half its error, traced forward again, limited by the source's four
    texels around the backward trace; then the dissipation, the evaporation and the guard. Draw into the field."""

    def use(self, source: Texture, forward: Texture, velocity: Texture, obstacle: Texture, aspect: float, timestep: float,
            dissipation: float, has_obstacles: bool = True, evaporation: float = 0.0, evaporation_knee: float = 0.1,
            measure: int = 0, guard: GuardSlot | None = None, clamp: bool = True) -> None:
        """Write the corrected step into the bound target.

        Args:
            source: The field before the step
            forward: The semi-Lagrangian predictor, the step without dissipation or evaporation
            velocity: The velocity the step traced through (RG16F), widths/s
            obstacle: Obstacle mask (R8)
            aspect: The field's width over its height: one width is this share of the height
            timestep: dt × advection_scale
            dissipation: The fade's multiplier this step; 1 = none
            has_obstacles: Whether obstacle logic is active
            evaporation: dt / evaporation_time, 0 off
            evaporation_knee: The size below which the loss turns from a fixed amount to a share
            measure: How the size is read: 0 a premultiplied density's alpha, 1 a vector's length, 2 |r|
            guard: The field's slot in its stage's FiniteGuard; None no guard
            clamp: Where the correction overshoots, cut it to the bounds instead of reverting to the predictor
        """
        if not self.allocated or not self.shader_program:
            return
        glUseProgram(self.shader_program)
        for unit, (texture, name) in enumerate(((velocity, "uVelocity"), (source, "uSource"), (forward, "uForward"),
                                                (obstacle, "uObstacle"))):
            glActiveTexture((GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE2, GL_TEXTURE3)[unit])
            glBindTexture(GL_TEXTURE_2D, texture.tex_id)
            glUniform1i(self.get_uniform_loc(name), unit)
        glUniform1i(self.get_uniform_loc("uHasObstacles"), int(has_obstacles))
        glUniform1f(self.get_uniform_loc("uTimestep"), timestep)
        glUniform2f(self.get_uniform_loc("uRdx"), 1.0, aspect if aspect > 0.0 else 1.0)
        glUniform1f(self.get_uniform_loc("uDissipation"), dissipation)
        glUniform1f(self.get_uniform_loc("uEvaporation"), evaporation)
        glUniform1f(self.get_uniform_loc("uEvaporationKnee"), evaporation_knee)
        glUniform1i(self.get_uniform_loc("uMeasure"), measure)
        glUniform1i(self.get_uniform_loc("uClamp"), int(clamp))
        if guard is not None:
            guard.use(self.get_uniform_loc("uGuardSlot"))
        else:
            glUniform1i(self.get_uniform_loc("uGuardSlot"), -1)
        draw_quad()
