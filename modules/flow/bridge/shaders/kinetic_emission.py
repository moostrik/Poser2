"""Kinetic Emission shader.

Turns the held release into the emission the density bridge reads.
"""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class KineticEmission(Shader):
    """Writes held release * gain to the red channel."""

    def use(self, held_release: Texture, gain: float) -> None:
        """Compose the emission.

        Args:
            held_release: Held release in the red channel
            gain: Emission gain
        """
        if not self.allocated or not self.shader_program:
            return
        if not held_release.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, held_release.tex_id)

        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform1f(self.get_uniform_loc("gain"), gain)

        draw_quad()
