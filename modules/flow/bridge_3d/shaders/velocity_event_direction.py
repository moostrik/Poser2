"""Velocity Event Direction shader.

Turns an event into a velocity along a trail's direction.
"""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class VelocityEventDirection(Shader):
    """Writes the event along the trail's direction to RGB."""

    def use(self, trail: Texture, event: Texture) -> None:
        """Compose the event's velocity.

        Args:
            trail: RGB trail, whose direction the event takes
            event: The event in the red channel
        """
        if not self.allocated or not self.shader_program:
            return
        if not trail.allocated or not event.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, trail.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, event.tex_id)

        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform1i(self.get_uniform_loc("tex1"), 1)

        draw_quad()
