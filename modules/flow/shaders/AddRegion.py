"""Blend Region shader.

Adds source texture to a specific region of destination.
Source is stretched to fill the target region.
"""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture


class AddRegion(Shader):
    """Add source texture to a specific region of destination."""

    def use(self, dst: Texture, src: Texture,
            x: float, y: float, w: float, h: float,
            src_strength: float = 1.0) -> None:
        """Render blended result with source added to region.

        Args:
            dst: Destination texture (passed through, added in region)
            src: Source texture (stretched to fill region)
            x: Region X position in UV coordinates [0, 1]
            y: Region Y position in UV coordinates [0, 1]
            w: Region width in UV coordinates
            h: Region height in UV coordinates
            src_strength: Multiplier for source texture
        """
        if not self.allocated or not self.shader_program:
            print("BlendRegion shader not allocated or shader program missing.")
            return
        if not dst.allocated or not src.allocated:
            print("BlendRegion shader: input textures not allocated.")
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, dst.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, src.tex_id)

        glUniform1i(self.get_uniform_loc("dst"), 0)
        glUniform1i(self.get_uniform_loc("src"), 1)
        glUniform1f(self.get_uniform_loc("src_strength"), src_strength)
        glUniform4f(self.get_uniform_loc("region"), x, y, w, h)

        draw_quad()
