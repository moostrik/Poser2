"""Kinetic Onset shader.

Turns the onset, the positive acceleration, into a velocity along the motion.
"""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class KineticOnset(Shader):
    """Writes max(0, acceleration) * strength along the fast trail to RGB."""

    def use(self, fast: Texture, acceleration: Texture, strength: float) -> None:
        """Compose the onset force.

        Args:
            fast: RGB fast trail
            acceleration: Signed acceleration in the red channel
            strength: Onset force
        """
        if not self.allocated or not self.shader_program:
            return
        if not fast.allocated or not acceleration.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, fast.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, acceleration.tex_id)

        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform1i(self.get_uniform_loc("tex1"), 1)
        glUniform1f(self.get_uniform_loc("strength"), strength)

        draw_quad()
