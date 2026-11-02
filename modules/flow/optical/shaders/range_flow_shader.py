"""Range flow shader: one sampling stride of the motion from two smoothed depths."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE2, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class RangeFlowShader(Shader):
    """Writes the displacement so far in uv (RG), the velocity toward the camera in m/s (B) and the share of the window that counted (A)."""

    def use(self, previous: Texture, current: Texture, estimate: Texture, stride: int, window: int, regularizer: float,
            jump: float, dt: float) -> None:
        """Refine `estimate` at one sampling stride.

        Args:
            previous: Previous smoothed depth, metres, 0 for none
            current: Current smoothed depth, metres, 0 for none
            estimate: The motion from the wider strides
            stride: Texels between the window's samples
            window: Window radius in strides
            regularizer: Damps sideways motion where the depth is flat
            jump: More depth difference than this is another surface (m)
            dt: Seconds between the two depths
        """
        if not self.allocated or not self.shader_program:
            return
        if not previous.allocated or not current.allocated or not estimate.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, previous.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, current.tex_id)
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, estimate.tex_id)

        glUniform1i(self.get_uniform_loc("previous"), 0)
        glUniform1i(self.get_uniform_loc("current"), 1)
        glUniform1i(self.get_uniform_loc("estimate"), 2)
        glUniform1i(self.get_uniform_loc("stride"), stride)
        glUniform1i(self.get_uniform_loc("window"), window)
        glUniform1f(self.get_uniform_loc("regularizer"), regularizer)
        glUniform1f(self.get_uniform_loc("jump"), jump)
        glUniform1f(self.get_uniform_loc("dt"), max(dt, 1e-4))

        draw_quad()
