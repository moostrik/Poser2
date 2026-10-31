"""Kinetic Input shader.

Copies the optical flow into the kinetic bridge without its non-finite pixels.
"""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class KineticInput(Shader):
    """Writes the flow to RG, zero where it is not finite."""

    def use(self, flow: Texture) -> None:
        """Copy the flow.

        Args:
            flow: RG optical flow
        """
        if not self.allocated or not self.shader_program:
            return
        if not flow.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, flow.tex_id)

        glUniform1i(self.get_uniform_loc("tex0"), 0)

        draw_quad()
