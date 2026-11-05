"""Displacement filter shader: x/y through the depth track's filter, so it lags as z does."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE2, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class DisplacementFilter(Shader):
    """Writes the filtered displacement (RG, uv per frame) and the position error (BA, uv)."""

    def use(self, displacement: Texture, state: Texture, carry: Texture, alpha: float) -> None:
        """Filter one frame.

        Args:
            displacement: The measured displacement, uv per frame
            state: The filter the frame before
            carry: The displacement the state is carried by, uv per frame
            alpha: Weight of the measurement on the position (1 takes it whole)
        """
        if not self.allocated or not self.shader_program:
            return
        if not displacement.allocated or not state.allocated or not carry.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, displacement.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, state.tex_id)
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, carry.tex_id)

        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform1i(self.get_uniform_loc("tex1"), 1)
        glUniform1i(self.get_uniform_loc("carry"), 2)
        glUniform1f(self.get_uniform_loc("alpha"), alpha)

        draw_quad()
