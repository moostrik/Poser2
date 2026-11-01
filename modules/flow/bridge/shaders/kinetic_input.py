"""Kinetic Input shader.

Copies the optical flow and the velocity toward the camera into the kinetic bridge without non-finite pixels.
"""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class KineticInput(Shader):
    """Writes (flow x, flow y, velocity toward the camera) to RGB, zero where it is not finite."""

    def use(self, flow: Texture, depth_velocity: Texture | None) -> None:
        """Copy the velocity.

        Args:
            flow: RG optical flow
            depth_velocity: Velocity toward the camera in red, None without depth
        """
        if not self.allocated or not self.shader_program:
            return
        if not flow.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, flow.tex_id)
        if depth_velocity is not None:
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, depth_velocity.tex_id)
            glUniform1i(self.get_uniform_loc("tex1"), 1)

        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform1i(self.get_uniform_loc("has_depth"), 1 if depth_velocity is not None else 0)

        draw_quad()
