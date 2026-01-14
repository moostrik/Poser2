"""Gaussian Blur shader.

Two-pass separable Gaussian blur for spatial smoothing.
Ported from ofxFlowTools ftGaussianBlurShader.h
"""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture, Fbo


class GaussianBlur(Shader):
    """Separable Gaussian blur shader.

    Uses binomial weights for efficient blur.
    Requires two passes (horizontal + vertical) for full 2D blur.
    """

    def __init__(self) -> None:
        super().__init__()
        self._temp_fbo: Fbo = Fbo()

    def allocate(self) -> None:
        """Allocate shader resources."""
        super().allocate()

    def deallocate(self) -> None:
        """Deallocate shader resources."""
        super().deallocate()
        if self._temp_fbo.allocated:
            self._temp_fbo.deallocate()

    def use(self, source: Texture, radius: float, target: Fbo) -> None:
        """Apply two-pass blur (horizontal + vertical).

        Args:
            source: Source texture to blur
            radius: Blur radius in pixels
            target: Target FBO for output
        """
        if not self.allocated or not self.shader_program:
            return
        if not source.allocated or not target.allocated:
            return
        if radius <= 0:
            return

        # Allocate temp FBO if needed
        if not self._temp_fbo.allocated:
            self._temp_fbo.allocate(source.width, source.height, source.internal_format)

        # Horizontal pass to temp
        self._temp_fbo.begin()
        self._blur_pass(source, radius, horizontal=True)
        self._temp_fbo.end()

        # Vertical pass to target
        target.begin()
        self._blur_pass(self._temp_fbo, radius, horizontal=False)
        target.end()

    def _blur_pass(self, source: Texture, radius: float, horizontal: bool) -> None:
        """Single blur pass (horizontal or vertical).

        Args:
            source: Source texture
            radius: Blur radius
            horizontal: True for horizontal pass, False for vertical
        """
        glUseProgram(self.shader_program)

        # Bind texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, source.tex_id)

        # Set uniforms
        glUniform1i(glGetUniformLocation(self.shader_program, "tex0"), 0)
        glUniform1f(glGetUniformLocation(self.shader_program, "radius"), radius)
        glUniform1i(glGetUniformLocation(self.shader_program, "horizontal"), 1 if horizontal else 0)

        # Texel size for normalized offset calculation
        glUniform2f(
            glGetUniformLocation(self.shader_program, "texel_size"),
            1.0 / source.width,
            1.0 / source.height
        )

        draw_quad()

        # Cleanup
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glUseProgram(0)
