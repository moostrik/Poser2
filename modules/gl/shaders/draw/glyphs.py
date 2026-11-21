"""Text rendering shader.

Draws a whole string's glyph quads from a font atlas in one call: the caller uploads the
batch (positions in 0..1 of the target, origin top-left, + atlas UVs, 6 vertices per glyph) into its own VAO
and binds it.
One draw per string instead of one per glyph keeps the main thread's Python/GL step count
— and with it its GIL exposure — flat in the text length.
"""

from OpenGL.GL import GL_TRIANGLES, glBindTextureUnit, glDrawArrays, glUniform4f, glUseProgram
from modules.utils import Color
from ...shader import Shader


class Glyphs(Shader):
    """Shader for rendering a batch of text glyphs from a font atlas."""

    def use(self, vertex_count: int,
            color: Color,
            atlas_texture_id: int) -> None:
        """Draw ``vertex_count`` vertices of pre-uploaded glyph quads.

        Args:
            vertex_count: Number of vertices in the bound VAO (6 per glyph)
            color: the text's color
            atlas_texture_id: OpenGL texture ID of font atlas
        """
        if not self.allocated or not self.shader_program or vertex_count <= 0:
            return

        glUseProgram(self.shader_program)

        glBindTextureUnit(0, atlas_texture_id)

        glUniform4f(self.get_uniform_loc("color"), *color)

        glDrawArrays(GL_TRIANGLES, 0, vertex_count)
