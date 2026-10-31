"""Kinetic Hold shader.

Peak-holds the release with an exponential decay.
"""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class KineticHold(Shader):
    """Writes max(release, previous * decay) to the red channel."""

    def use(self, acceleration: Texture, previous: Texture, decay: float) -> None:
        """Hold the release.

        Args:
            acceleration: Signed acceleration in the red channel
            previous: Previous held release
            decay: What remains of the held release this update
        """
        if not self.allocated or not self.shader_program:
            return
        if not acceleration.allocated or not previous.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, acceleration.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, previous.tex_id)

        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform1i(self.get_uniform_loc("tex1"), 1)
        glUniform1f(self.get_uniform_loc("decay"), decay)

        draw_quad()
