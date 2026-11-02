"""Compose Velocity shader: a flow's x/y and z as one 3D velocity."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1i, glUniform3f, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class ComposeVelocity(Shader):
    """Writes (x, y, z) times strength to RGB, zero where it is not finite."""

    def use(self, xy: Texture, z: Texture | None, strength: tuple[float, float, float]) -> None:
        """Compose the velocity.

        Args:
            xy: x and y in RG
            z: z in R, None without depth (z is 0)
            strength: Multiplier on x, y and z
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

        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform1i(self.get_uniform_loc("has_depth"), 1 if z is not None else 0)
        glUniform3f(self.get_uniform_loc("strength"), *strength)

        draw_quad()
