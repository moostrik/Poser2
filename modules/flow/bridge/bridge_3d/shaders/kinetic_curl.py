"""Kinetic Curl shader.

The 2D curl of the fast trail, the extra input to the fluid's vorticity force.
"""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class KineticCurl(Shader):
    """Writes the 2D curl of a velocity field to the red channel."""

    def use(self, velocity: Texture) -> None:
        """Compute the curl.

        Args:
            velocity: RG velocity field
        """
        if not self.allocated or not self.shader_program:
            return
        if not velocity.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, velocity.tex_id)

        glUniform1i(self.get_uniform_loc("tex0"), 0)

        draw_quad()
