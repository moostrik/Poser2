"""Strip Roll shader.

Rolls the strip one column and draws the new one from the summary's top mip, one lane per channel.
"""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class StripRoll(Shader):
    """Writes the previous strip moved one column left, with the new column at the right."""

    def use(self, previous: Texture, summary: Texture, scale: float) -> None:
        """Advance the strip.

        Args:
            previous: Previous strip
            summary: Summary texture with mipmaps, its top mip the mean amount per channel
            scale: Multiplier on the bars
        """
        if not self.allocated or not self.shader_program:
            return
        if not previous.allocated or not summary.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, previous.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, summary.tex_id)

        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform1i(self.get_uniform_loc("tex1"), 1)
        glUniform1f(self.get_uniform_loc("scale"), scale)

        draw_quad()
