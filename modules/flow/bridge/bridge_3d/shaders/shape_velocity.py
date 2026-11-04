"""Shape Velocity shader: the trail bridge's power curve and limit on a velocity's x/y, then its scale per axis."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUniform3f, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class ShapeVelocity(Shader):
    """Writes the velocity with its x/y length to `power`, rolled off softly toward `limit`, then x, y, z times `scale` (RGB)."""

    def use(self, velocity: Texture, power: float, limit: float, scale: tuple[float, float, float]) -> None:
        """Shape `velocity`.

        Args:
            velocity: x, y, z in RGB
            power: Power curve on the x/y length
            limit: The x/y length the roll-off approaches
            scale: Multiplier on x, y and z, after the shaping
        """
        if not self.allocated or not self.shader_program or not velocity.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, velocity.tex_id)

        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform1f(self.get_uniform_loc("power"), power)
        glUniform1f(self.get_uniform_loc("limit"), max(limit, 1e-3))
        glUniform3f(self.get_uniform_loc("scale"), *scale)

        draw_quad()
