"""Compose Velocity 2D shader: a flow's x/y as one 2D velocity."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1i, glUniform2f, glUseProgram
from modules.gl import Shader, draw_quad, Texture
from ...finite_guard import GuardSlot


class ComposeVelocity2D(Shader):
    """Writes (x, y) times strength to RG, times the mask when given; 0 where it is not finite, counted by the
    guard."""

    def use(self, xy: Texture, strength: tuple[float, float], mask: Texture | None = None,
            guard: GuardSlot | None = None) -> None:
        """Compose the velocity.

        Args:
            xy: x and y in RG
            strength: Multiplier on x and y
            mask: Multiplier in R (0 to 1) on x and y, None for no mask
            guard: The velocity's slot in the motion's FiniteGuard; None no guard
        """
        if not self.allocated or not self.shader_program or not xy.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, xy.tex_id)
        if mask is not None:
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, mask.tex_id)
            glUniform1i(self.get_uniform_loc("tex1"), 1)

        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform1i(self.get_uniform_loc("has_mask"), 1 if mask is not None else 0)
        glUniform2f(self.get_uniform_loc("strength"), *strength)
        if guard is not None:
            guard.use(self.get_uniform_loc("uGuardSlot"))
        else:
            glUniform1i(self.get_uniform_loc("uGuardSlot"), -1)

        draw_quad()
