"""Histogram Count shader.

Counts a texture's four channels into the bins of an R32UI texture, one row per channel; or its xyz's length
into the first row's bins on a log scale.
"""

from OpenGL.GL import GL_R32UI, glUniform1f, glUniform1i, glUniform2i, glUseProgram
from modules.gl import ComputeShader, Texture


class HistogramCount(ComputeShader):
    """Adds the values of a grid of points of a texture to its bins; clear them first."""

    def use(self, texture: Texture, histogram: Texture, scale: float, samples: tuple[int, int],
            speed: tuple[float, float] | None = None) -> None:
        """Count the texture.

        Args:
            texture: The texture counted
            histogram: R32UI bins, its width the bin count, one row per channel
            scale: Multiplier on the values; the bins run from -1 to 1 after it
            samples: The grid of samples, across and down
            speed: The log scale's edges, low and high, to count the length of xyz instead into the first row;
                None for the channels
        """
        if not self.allocated or not self.shader_program:
            return
        if not texture.allocated or not histogram.allocated:
            return

        glUseProgram(self.shader_program)
        texture.bind(0)
        self.bind_image_readwrite(0, histogram, GL_R32UI)
        glUniform1f(self.get_uniform_loc("scale"), scale)
        glUniform2i(self.get_uniform_loc("samples"), *samples)
        glUniform1i(self.get_uniform_loc("speed"), int(speed is not None))
        if speed is not None:
            glUniform1f(self.get_uniform_loc("speed_low"), speed[0])
            glUniform1f(self.get_uniform_loc("speed_high"), speed[1])
        self.dispatch(*samples)
        self.barrier_texture_fetch()
