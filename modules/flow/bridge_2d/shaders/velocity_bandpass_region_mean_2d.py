"""Velocity Bandpass Region Mean 2D shader.

Writes a region's mean velocity over its moving pixels, one texel per region.
"""

from OpenGL.GL import glUniform1f, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class VelocityBandpassRegionMean2D(Shader):
    """Writes the region's sums over its moving share to RG."""

    def use(self, regions: Texture, lod: float) -> None:
        """Write the means.

        Args:
            regions: velocity * moving in RG, moving in B, with mipmaps
            lod: Mip level of a region
        """
        if not self.allocated or not self.shader_program or not regions.allocated:
            return
        glUseProgram(self.shader_program)
        regions.bind(0)
        glUniform1f(self.get_uniform_loc("lod"), lod)
        draw_quad()
