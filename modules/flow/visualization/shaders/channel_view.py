"""Channel View shader.

Draws a texture's channels as an image or as a colormap centered on mid grey, all of them or one.
"""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class ChannelView(Shader):
    """Maps a texture's channels to a color: as stored, or as a colormap centered on mid grey."""

    def use(self, texture: Texture, scale: float, channel: int, colormap: bool, vector: bool = False) -> None:
        """Draw the texture.

        Args:
            texture: The texture shown
            scale: Multiplier on the values
            channel: 0 to 3 for x, y, z, w; 4 for all
            colormap: Centered on mid grey, x, y, z as red, green, blue and w as grey, instead of as stored
            vector: With colormap, the color its direction's and its brightness its size, black at 0
        """
        if not self.allocated or not self.shader_program:
            return
        if not texture.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture.tex_id)

        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform1f(self.get_uniform_loc("scale"), scale)
        glUniform1i(self.get_uniform_loc("channel"), channel)
        glUniform1i(self.get_uniform_loc("colormap"), 1 if colormap else 0)
        glUniform1i(self.get_uniform_loc("is_vector"), 1 if vector else 0)

        draw_quad()
