"""Shape Velocity shader: the velocity trail bridge's soft limit on z."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class ShapeVelocity(Shader):
    """Writes the velocity with x/y as it is and z rolled off toward `z_limit` (RGB)."""

    def use(self, velocity: Texture, z_limit: float) -> None:
        """Shape `velocity`.

        Args:
            velocity: x, y, z in RGB, widths/s
            z_limit: The z the roll-off approaches
        """
        if not self.allocated or not self.shader_program or not velocity.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, velocity.tex_id)

        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform1f(self.get_uniform_loc("z_limit"), max(z_limit, 1e-3))

        draw_quad()
