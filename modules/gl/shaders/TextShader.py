"""Text rendering shader.

Renders textured quads for font glyphs using a font atlas texture.
"""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad


class TextShader(Shader):
    """Shader for rendering text glyphs from a font atlas."""

    def use(self, glyph_rect: tuple[float, float, float, float],
            uv_rect: tuple[float, float, float, float],
            text_color: tuple[float, float, float, float],
            screen_size: tuple[float, float],
            atlas_texture_id: int) -> None:
        """Render a single glyph quad.

        Args:
            glyph_rect: (x, y, width, height) in screen pixels
            uv_rect: (u0, v0, u1, v1) texture coordinates
            text_color: (r, g, b, a) text color
            screen_size: (width, height) of render target
            atlas_texture_id: OpenGL texture ID of font atlas
        """
        if not self.allocated or not self.shader_program:
            return

        glUseProgram(self.shader_program)

        # Bind atlas texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, atlas_texture_id)
        glUniform1i(self.get_uniform_loc("atlas"), 0)

        # Set uniforms
        glUniform2f(self.get_uniform_loc("screen_size"), *screen_size)
        glUniform4f(self.get_uniform_loc("glyph_rect"), *glyph_rect)
        glUniform4f(self.get_uniform_loc("uv_rect"), *uv_rect)
        glUniform4f(self.get_uniform_loc("text_color"), *text_color)

        # Draw quad
        draw_quad()
