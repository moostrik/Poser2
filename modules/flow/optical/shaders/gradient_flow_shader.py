"""Gradient Flow shader.

The motion along the brightness gradient between two luminance frames.
Ported from ofxFlowTools ftOpticalFlowShader.h
"""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUniform2f, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class GradientFlowShader(Shader):
    """Writes the x/y motion strength (RG), times strength, the threshold taken off its length."""

    def use(self, previous: Texture, current: Texture, offset: int, threshold: float, x_gain: float, y_gain: float) -> None:
        """Measure between two frames.

        Args:
            previous: Previous luminance frame, mipmapped
            current: Current luminance frame, mipmapped
            offset: Gradient sample offset in pixels
            threshold: Taken off the length, after strength
            x_gain: X multiplier (negative inverts)
            y_gain: Y multiplier (negative inverts)
        """
        if not self.allocated or not self.shader_program:
            return
        if not previous.allocated or not current.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, previous.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, current.tex_id)

        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform1i(self.get_uniform_loc("tex1"), 1)
        glUniform2f(self.get_uniform_loc("offset"), float(offset) / previous.width, float(offset) / previous.height)
        glUniform1f(self.get_uniform_loc("threshold"), threshold)
        glUniform2f(self.get_uniform_loc("gain"), x_gain, y_gain)

        draw_quad()
