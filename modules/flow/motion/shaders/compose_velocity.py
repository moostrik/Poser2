"""Compose Velocity shader: a flow's x/y and z as one 3D velocity."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE2, GL_TEXTURE3, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1i, glUniform3f, glUseProgram
from modules.gl import Shader, draw_quad, Texture
from ...finite_guard import GuardSlot


class ComposeVelocity(Shader):
    """Writes (x, y, z) times strength to RGB, times the mask when given, and z also times the depth mask when
    given; 0 where it is not finite, counted by the guard."""

    def use(self, xy: Texture, z: Texture | None, strength: tuple[float, float, float], mask: Texture | None = None,
            guard: GuardSlot | None = None, depth_mask: Texture | None = None) -> None:
        """Compose the velocity.

        Args:
            xy: x and y in RG
            z: z in R, None without depth (z is 0)
            strength: Multiplier on x, y and z
            mask: Multiplier in R (0 to 1) on x, y and z, None for no mask
            guard: The velocity's slot in the motion's FiniteGuard; None no guard
            depth_mask: Multiplier in R (0 to 1) on z alone, where the depth is the player's; None for none
        """
        if not self.allocated or not self.shader_program or not xy.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, xy.tex_id)
        if z is not None:
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, z.tex_id)
            glUniform1i(self.get_uniform_loc("tex1"), 1)
        if mask is not None:
            glActiveTexture(GL_TEXTURE2)
            glBindTexture(GL_TEXTURE_2D, mask.tex_id)
            glUniform1i(self.get_uniform_loc("tex2"), 2)
        if depth_mask is not None:
            glActiveTexture(GL_TEXTURE3)
            glBindTexture(GL_TEXTURE_2D, depth_mask.tex_id)
            glUniform1i(self.get_uniform_loc("tex3"), 3)

        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform1i(self.get_uniform_loc("has_depth"), 1 if z is not None else 0)
        glUniform1i(self.get_uniform_loc("has_mask"), 1 if mask is not None else 0)
        glUniform1i(self.get_uniform_loc("has_depth_mask"), 1 if depth_mask is not None else 0)
        glUniform3f(self.get_uniform_loc("strength"), *strength)
        if guard is not None:
            guard.use(self.get_uniform_loc("uGuardSlot"))
        else:
            glUniform1i(self.get_uniform_loc("uGuardSlot"), -1)

        draw_quad()
