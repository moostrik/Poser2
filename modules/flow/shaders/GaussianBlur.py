"""Gaussian Blur shader.

Single-pass separable Gaussian blur for spatial smoothing.
Optimized implementation using linear sampling.
"""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture


class GaussianBlur(Shader):
    """Separable Gaussian blur shader.

    Single-pass blur (horizontal OR vertical).
    Caller manages two-pass ping-pong with SwapFbo.
    Uses optimized linear sampling for better performance.
    """

    def use(self, source_texture: Texture, radius: float, horizontal: bool) -> None:
        """Apply single blur pass (horizontal or vertical).

        Args:
            source_texture: Source texture to blur
            radius: Blur radius in pixels (0-10 recommended)
            horizontal: True for horizontal pass, False for vertical
        """
        if not self.allocated or not self.shader_program:
            print("GaussianBlur shader not allocated or shader program missing.")
            return
        if not source_texture.allocated:
            print("GaussianBlur shader: input texture not allocated.")
            return
        if radius <= 0:
            # No blur, just draw source
            source_texture.draw(0, 0, source_texture.width, source_texture.height)
            return

        glUseProgram(self.shader_program)

        # Bind texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, source_texture.tex_id)
        glUniform1i(self.get_uniform_loc("tex0"), 0)

        # Set uniforms (using cached locations)
        glUniform1f(self.get_uniform_loc("radius"), radius)
        glUniform1i(self.get_uniform_loc("horizontal"), 1 if horizontal else 0)
        glUniform2f(self.get_uniform_loc("resolution"), float(source_texture.width), float(source_texture.height))

        draw_quad()
