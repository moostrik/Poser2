"""Velocity Trail shader: one Ema step of the velocity trail bridge, x/y with an attack."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class VelocityTrail(Shader):
    """Writes weight * previous + (1 - weight) * velocity (RGB): x/y by `attack` where its speed rises, else by `release`; z by `release`."""

    def use(self, previous: Texture, velocity: Texture, attack: float, release: float) -> None:
        """Advance the trail.

        Args:
            previous: The trail so far, xyz in RGB
            velocity: The shaped velocity, xyz in RGB
            attack: What remains of the trail where the x/y speed rises
            release: What remains elsewhere, and of z
        """
        if not self.allocated or not self.shader_program or not previous.allocated or not velocity.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, previous.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, velocity.tex_id)

        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform1i(self.get_uniform_loc("tex1"), 1)
        glUniform1f(self.get_uniform_loc("attack"), attack)
        glUniform1f(self.get_uniform_loc("release"), release)

        draw_quad()
