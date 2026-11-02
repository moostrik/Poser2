"""Displacement velocity shader: a thresholded displacement shaped as the trail bridge's velocity."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUniform2f, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class DisplacementVelocity(Shader):
    """Writes the shaped velocity (RG)."""

    def use(self, displacement: Texture, scale: float, strength_x: float, strength_y: float, power: float) -> None:
        """Shape `displacement` (uv per frame, noise floor already taken off).

        Args:
            displacement: uv per frame
            scale: Velocity per pixel of displacement
            strength_x: X multiplier (negative inverts)
            strength_y: Y multiplier (negative inverts)
            power: Power curve on the length
        """
        if not self.allocated or not self.shader_program or not displacement.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, displacement.tex_id)

        glUniform1i(self.get_uniform_loc("displacement"), 0)
        glUniform1f(self.get_uniform_loc("scale"), scale)
        glUniform2f(self.get_uniform_loc("force"), strength_x, strength_y)
        glUniform1f(self.get_uniform_loc("power"), power)

        draw_quad()
