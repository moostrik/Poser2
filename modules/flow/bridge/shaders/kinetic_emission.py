"""Kinetic Emission shader.

Turns the held release into the emission the density bridge reads.
"""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class KineticEmission(Shader):
    """Writes held release * gain, times the reservoir when it is used, to the red channel."""

    def use(self, held_release: Texture, reservoir: Texture, gain: float, use_reservoir: bool) -> None:
        """Compose the emission.

        Args:
            held_release: Held release in the red channel
            reservoir: Reservoir in the red channel
            gain: Emission gain
            use_reservoir: Whether the reservoir scales the emission
        """
        if not self.allocated or not self.shader_program:
            return
        if not held_release.allocated or not reservoir.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, held_release.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, reservoir.tex_id)

        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform1i(self.get_uniform_loc("tex1"), 1)
        glUniform1f(self.get_uniform_loc("gain"), gain)
        glUniform1i(self.get_uniform_loc("use_reservoir"), 1 if use_reservoir else 0)

        draw_quad()
