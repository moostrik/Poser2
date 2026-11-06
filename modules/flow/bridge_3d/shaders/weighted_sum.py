"""Weighted Sum shader.

The bridge's mix: up to five sources, each times its weight, summed.
"""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE2, GL_TEXTURE3, GL_TEXTURE4, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1fv, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture

MAX_SOURCES: int = 5
UNITS: tuple[int, ...] = (GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE2, GL_TEXTURE3, GL_TEXTURE4)


class WeightedSum(Shader):
    """Writes the sum of each source times its weight to RGB."""

    def use(self, sources: list[tuple[Texture, float]]) -> None:
        """Mix the sources.

        Args:
            sources: Up to five (texture, weight) pairs; the RGB of each is taken times its weight
        """
        if not self.allocated or not self.shader_program or not sources or len(sources) > MAX_SOURCES:
            return
        if not all(texture.allocated for texture, _ in sources):
            return

        glUseProgram(self.shader_program)

        weights: list[float] = [0.0] * MAX_SOURCES
        for unit in range(MAX_SOURCES):
            texture, weight = sources[unit] if unit < len(sources) else (sources[0][0], 0.0)
            weights[unit] = weight
            glActiveTexture(UNITS[unit])
            glBindTexture(GL_TEXTURE_2D, texture.tex_id)
            glUniform1i(self.get_uniform_loc(f"tex{unit}"), unit)
        glUniform1fv(self.get_uniform_loc("weights"), MAX_SOURCES, weights)

        draw_quad()
