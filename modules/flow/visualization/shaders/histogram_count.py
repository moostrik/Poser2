"""Histogram Count shader.

Counts a texture's four channels into the bins of an R32UI texture, one row per channel.
"""

from OpenGL.GL import GL_R32UI, glUniform1f, glUniform2i, glUseProgram
from modules.gl import ComputeShader, Texture


class HistogramCount(ComputeShader):
    """Adds the values of a grid of points of a texture to its bins; clear them first."""

    def use(self, texture: Texture, histogram: Texture, scale: float, samples: tuple[int, int]) -> None:
        """Count the texture.

        Args:
            texture: The texture counted
            histogram: R32UI bins, its width the bin count, one row per channel
            scale: Multiplier on the values; the bins run from -1 to 1 after it
            samples: The grid of samples, across and down
        """
        if not self.allocated or not self.shader_program:
            return
        if not texture.allocated or not histogram.allocated:
            return

        glUseProgram(self.shader_program)
        self.bind_texture(0, texture, "tex0")
        self.bind_image_readwrite(0, histogram, GL_R32UI)
        glUniform1f(self.get_uniform_loc("scale"), scale)
        glUniform2i(self.get_uniform_loc("samples"), *samples)
        self.dispatch(*samples)
        self.barrier_texture_fetch()
