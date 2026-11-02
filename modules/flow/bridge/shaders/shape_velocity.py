"""Shape Velocity shader: the trail bridge's power curve and clamp on a velocity's x/y."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class ShapeVelocity(Shader):
    """Writes the velocity with its x/y length to `power`, clamped to ±1 (RGB, z unchanged)."""

    def use(self, velocity: Texture, power: float) -> None:
        """Shape `velocity`.

        Args:
            velocity: x, y, z in RGB
            power: Power curve on the x/y length
        """
        if not self.allocated or not self.shader_program or not velocity.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, velocity.tex_id)

        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform1f(self.get_uniform_loc("power"), power)

        draw_quad()
