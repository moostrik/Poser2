"""Speed Count 2D shader.

Counts a 2D velocity's speeds, the length of its xy, into the first row of an R32UI texture's bins on a log scale.
"""

from OpenGL.GL import GL_R32UI, glUniform1f, glUniform2i, glUseProgram
from modules.gl import ComputeShader, Texture

SPEED_DEFAULT: tuple[float, float] = (0.01, 10.0)     # widths/s, the log scale's edges without a range given


class SpeedCount2D(ComputeShader):
    """Adds the speeds of a grid of points of a 2D velocity to its bins; clear them first."""

    def use(self, texture: Texture, histogram: Texture, scale: float, samples: tuple[int, int],
            speed: tuple[float, float] | None = None) -> None:
        """Count the speeds.

        Args:
            texture: The velocity counted (xy)
            histogram: R32UI bins, its width the bin count
            scale: Multiplier on the values
            samples: The grid of samples, across and down
            speed: The log scale's edges, low and high
        """
        if not self.allocated or not self.shader_program:
            return
        if not texture.allocated or not histogram.allocated:
            return
        low, high = speed if speed is not None else SPEED_DEFAULT
        glUseProgram(self.shader_program)
        texture.bind(0)
        self.bind_image_readwrite(0, histogram, GL_R32UI)
        glUniform1f(self.get_uniform_loc("scale"), scale)
        glUniform2i(self.get_uniform_loc("samples"), *samples)
        glUniform1f(self.get_uniform_loc("speed_low"), low)
        glUniform1f(self.get_uniform_loc("speed_high"), high)
        self.dispatch(*samples)
        self.barrier_texture_fetch()
