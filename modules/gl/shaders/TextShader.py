"""Text rendering shader.

Draws a whole string's glyph quads from a font atlas in one call: the caller uploads the
batch (pixel positions + atlas UVs, 6 vertices per glyph) into its own VAO and binds it.
One draw per string instead of one per glyph keeps the render thread's Python/GL step count
— and with it its GIL exposure — flat in the text length.
"""

from OpenGL.GL import *  # type: ignore
from ..Shader import Shader


class TextShader(Shader):
    """Shader for rendering a batch of text glyphs from a font atlas."""

    def use(self, vertex_count: int,
            text_color: tuple[float, float, float, float],
            screen_size: tuple[float, float],
            atlas_texture_id: int) -> None:
        """Draw ``vertex_count`` vertices of pre-uploaded glyph quads.

        Args:
            vertex_count: Number of vertices in the bound VAO (6 per glyph)
            text_color: (r, g, b, a) text color
            screen_size: (width, height) of render target
            atlas_texture_id: OpenGL texture ID of font atlas
        """
        if not self.allocated or not self.shader_program or vertex_count <= 0:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, atlas_texture_id)
        glUniform1i(self.get_uniform_loc("atlas"), 0)

        glUniform2f(self.get_uniform_loc("screen_size"), *screen_size)
        glUniform4f(self.get_uniform_loc("text_color"), *text_color)

        glDrawArrays(GL_TRIANGLES, 0, vertex_count)
