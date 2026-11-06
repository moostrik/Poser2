"""Velocity Change Region Mean shader.

Writes a region's mean velocity over its moving pixels, one texel per region.
"""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class VelocityChangeRegionMean(Shader):
    """Writes the region's sums over its moving share to RGB."""

    def use(self, regions: Texture, lod: float) -> None:
        """Write the means.

        Args:
            regions: velocity * moving in RGB, moving in A, with mipmaps
            lod: Mip level of a region
        """
        if not self.allocated or not self.shader_program:
            return
        if not regions.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, regions.tex_id)

        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform1f(self.get_uniform_loc("lod"), lod)

        draw_quad()
