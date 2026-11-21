"""Weighted Sum shader.

The bridge's mix: up to seven sources, each times its weight, summed; or, for an amount, the length of
each source's xyz times its weight, summed.
"""

from OpenGL.GL import glUniform1fv, glUniform1i, glUniform1iv, glUseProgram
from modules.gl import Shader, draw_quad, Texture

MAX_SOURCES: int = 7


class WeightedSum(Shader):
    """Writes the sum of each source times its weight to RGB, or with `lengths` the sum of their lengths to all
    three; a scalar source gives its red."""

    def use(self, sources: list[tuple[Texture, float]], lengths: bool = False, capped: bool = False,
            scalars: int = 0) -> None:
        """Mix the sources.

        Args:
            sources: Up to seven (texture, weight) pairs; the RGB of each is taken times its weight
            lengths: Take each source's length of its RGB instead, for an amount
            capped: Hold the sum at 1, a presence
            scalars: How many of the sources, the last ones, are scalars (the mask), each giving its red
        """
        if not self.allocated or not self.shader_program or not sources or len(sources) > MAX_SOURCES:
            return
        if not all(texture.allocated for texture, _ in sources):
            return

        glUseProgram(self.shader_program)

        weights: list[float] = [0.0] * MAX_SOURCES
        scalar: list[int] = [0] * MAX_SOURCES
        for unit in range(MAX_SOURCES):
            texture, weight = sources[unit] if unit < len(sources) else (sources[0][0], 0.0)
            weights[unit] = weight
            scalar[unit] = int(len(sources) - scalars <= unit < len(sources))
            texture.bind(unit)
        glUniform1fv(self.get_uniform_loc("weights"), MAX_SOURCES, weights)
        glUniform1iv(self.get_uniform_loc("scalars"), MAX_SOURCES, scalar)
        glUniform1i(self.get_uniform_loc("lengths"), int(lengths))
        glUniform1i(self.get_uniform_loc("capped"), int(capped))

        draw_quad()
