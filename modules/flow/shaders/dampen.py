"""Dampen shader.

Exponential drag on magnitude excess above a threshold.
Values below threshold pass through unchanged. Values above are
smoothly pulled back toward the threshold each frame.
"""

from OpenGL.GL import *  # type: ignore
from modules.gl import Shader, draw_quad, Texture


class Dampen(Shader):
    """Dampen texture values above a threshold via exponential drag."""

    def use(self, src: Texture, threshold: float, dampen_factor: float,
            include_alpha: bool = False) -> None:
        """Dampen source texture values above threshold.

        Args:
            src: Source texture
            threshold: Magnitude threshold below which values are untouched
            dampen_factor: Precomputed decay multiplier for excess
                           (pow(0.01, dt / dampen_time); 1.0 = no effect)
            include_alpha: If True, compute magnitude from RGBA (density).
                           If False, compute from RGB only (velocity/temp).
        """
        if not self.allocated or not self.shader_program:
            return
        if not src.allocated:
            return

        glUseProgram(self.shader_program)

        # Bind texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, src.tex_id)
        glUniform1i(self.get_uniform_loc("src"), 0)

        # Set uniforms
        glUniform1f(self.get_uniform_loc("uThreshold"), threshold)
        glUniform1f(self.get_uniform_loc("uDampenFactor"), dampen_factor)
        glUniform1i(self.get_uniform_loc("uIncludeAlpha"), int(include_alpha))

        draw_quad()
