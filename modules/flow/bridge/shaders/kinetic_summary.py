"""Kinetic Summary shader.

Packs the fields the kinetic strip plots into one RGBA texture.
"""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE2, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class KineticSummary(Shader):
    """Writes (speed, onset, release, emission) to RGBA."""

    def use(self, fast: Texture, acceleration: Texture, emission: Texture) -> None:
        """Pack the fields.

        Args:
            fast: RGB fast trail
            acceleration: Signed acceleration in the red channel
            emission: Emission in the red channel
        """
        if not self.allocated or not self.shader_program:
            return
        if not fast.allocated or not acceleration.allocated or not emission.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, fast.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, acceleration.tex_id)
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, emission.tex_id)

        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform1i(self.get_uniform_loc("tex1"), 1)
        glUniform1i(self.get_uniform_loc("tex2"), 2)

        draw_quad()
