"""Shape Velocity shader: the velocity trail bridge's curve and limit on a velocity's x/y speed, and its limit on z."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class ShapeVelocity(Shader):
    """Writes the velocity with its x/y speed bent around `pivot` by `curve` and rolled off toward `limit`, z rolled off toward `z_limit` (RGB)."""

    def use(self, velocity: Texture, pivot: float, curve: float, limit: float, z_limit: float) -> None:
        """Shape `velocity`.

        Args:
            velocity: x, y, z in RGB, widths/s
            pivot: x/y speed the curve leaves as it is
            curve: Exponent on the x/y speed around the pivot
            limit: The x/y speed the roll-off approaches; 0 = no roll-off
            z_limit: The z the roll-off approaches
        """
        if not self.allocated or not self.shader_program or not velocity.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, velocity.tex_id)

        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform1f(self.get_uniform_loc("pivot"), max(pivot, 1e-3))
        glUniform1f(self.get_uniform_loc("curve"), curve)
        glUniform1f(self.get_uniform_loc("limit"), max(limit, 0.0))
        glUniform1f(self.get_uniform_loc("z_limit"), max(z_limit, 1e-3))

        draw_quad()
