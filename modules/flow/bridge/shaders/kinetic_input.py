"""Kinetic Input shader.

Copies the optical flow and the velocity toward the camera into the kinetic bridge without non-finite pixels.
"""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1i, glUniform2f, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class KineticInput(Shader):
    """Writes (flow x, flow y, velocity toward the camera) to RGB, zero where it is not finite."""

    def use(self, flow: Texture, depth_velocity: Texture | None, flow_scale: tuple[float, float] = (1.0, 1.0)) -> None:
        """Copy the velocity.

        Args:
            flow: RG flow: the shaped optical flow, or a displacement in uv per frame
            depth_velocity: Velocity toward the camera in red, None without depth
            flow_scale: Multiplier on the flow's x and y; turns a displacement into image widths per second
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
        glUniform2f(self.get_uniform_loc("flow_scale"), flow_scale[0], flow_scale[1])

        draw_quad()
