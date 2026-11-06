"""Velocity Change Direction shader.

Turns a change into a velocity along a trail's direction.
"""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class VelocityBandpassDirection(Shader):
    """Writes the change along the trail's direction to RGB."""

    def use(self, trail: Texture, change: Texture) -> None:
        """Compose the change's velocity.

        Args:
            trail: RGB trail, whose direction the change takes
            change: The change in the red channel
        """
        if not self.allocated or not self.shader_program:
            return
        if not trail.allocated or not change.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, trail.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, change.tex_id)

        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform1i(self.get_uniform_loc("tex1"), 1)

        draw_quad()
