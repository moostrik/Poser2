"""DensityColorize shader - Map density channels to track colors."""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture


class DensityColorize(Shader):
    """Map RGBA density channels to corresponding track colors."""

    def __init__(self) -> None:
        super().__init__()

    def use(self, density: Texture, colors: list[tuple[float, float, float, float]]) -> None:
        """Colorize density channels with track colors.

        Args:
            density: Density field (RGBA16F) where each channel represents a track
            colors: List of up to 4 RGBA colors, one per channel/track
        """
        # Bind shader program
        glUseProgram(self.shader_program)

        # Bind density texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, density.tex_id)
        glUniform1i(self.get_uniform_loc("uDensity"), 0)

        # Set color uniforms (pad to 4 if fewer provided)
        padded_colors = list(colors) + [(0.0, 0.0, 0.0, 0.0)] * (4 - len(colors))
        for i, color in enumerate(padded_colors[:4]):
            glUniform4f(self.get_uniform_loc(f"uColors[{i}]"), *color)

        # Draw fullscreen quad
        draw_quad()
