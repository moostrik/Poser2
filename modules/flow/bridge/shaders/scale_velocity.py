"""Scale Velocity shader: a velocity's x, y and z, each times its scale."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1i, glUniform3f, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class ScaleVelocity(Shader):
    """Writes x, y, z times `scale` (RGB)."""

    def use(self, velocity: Texture, scale: tuple[float, float, float]) -> None:
        """Scale `velocity`.

        Args:
            velocity: x, y, z in RGB
            scale: Multiplier on x, y and z
        """
        if not self.allocated or not self.shader_program or not velocity.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, velocity.tex_id)

        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform3f(self.get_uniform_loc("scale"), *scale)

        draw_quad()
