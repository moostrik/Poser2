"""Histogram Count Volume shader.

Counts a volume's four channels into the bins of an R32UI texture, one row per channel.
"""

from OpenGL.GL import GL_R32UI, glUniform1f, glUniform3i, glUseProgram
from modules.gl import ComputeShader, Texture, Texture3D


class HistogramCountVolume(ComputeShader):
    """Adds the values of a 3D grid of points of a volume to its bins; clear them first."""

    WORKGROUP_SIZE_X = 16
    WORKGROUP_SIZE_Y = 16
    WORKGROUP_SIZE_Z = 1

    def use(self, volume: Texture3D, histogram: Texture, scale: float, samples: tuple[int, int, int]) -> None:
        """Count the volume.

        Args:
            volume: The volume counted
            histogram: R32UI bins, its width the bin count, one row per channel
            scale: Multiplier on the values; the bins run from -1 to 1 after it
            samples: The grid of samples, across, down and deep
        """
        if not self.allocated or not self.shader_program:
            return
        if not volume.allocated or not histogram.allocated:
            return

        glUseProgram(self.shader_program)
        self.bind_texture_3d(0, volume, "tex0")
        self.bind_image_readwrite(0, histogram, GL_R32UI)
        glUniform1f(self.get_uniform_loc("scale"), scale)
        glUniform3i(self.get_uniform_loc("samples"), *samples)
        self.dispatch(*samples)
        self.barrier_texture_fetch()
