"""RGB to Luminance shader.

Converts RGB image to single-channel luminance for optical flow input.
Ported from ofxFlowTools ftRGB2LuminanceShader.h
"""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad_pixels
from modules.gl import Fbo, Texture


class Luminance(Shader):
    """Convert RGB texture to luminance (grayscale) with optional Y-flip for Image textures."""

    def use(self, target_fbo: Fbo, source_tex: Texture, flip_y: bool = False) -> None:
        """Render luminance to FBO.

        Args:
            target_fbo: Target framebuffer
            source_tex: Source RGB texture
            flip_y: Flip texture coordinates vertically (True for Image textures)
        """
        if not self.allocated or not self.shader_program: return
        if not target_fbo.allocated or not source_tex.allocated: return

        # Activate shader program
        glUseProgram(self.shader_program)

        # Set up render target
        glBindFramebuffer(GL_FRAMEBUFFER, target_fbo.fbo_id)
        glViewport(0, 0, target_fbo.width, target_fbo.height)

        # Bind input texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, source_tex.tex_id)

        # Configure shader uniforms
        glUniform2f(glGetUniformLocation(self.shader_program, "resolution"), float(target_fbo.width), float(target_fbo.height))
        glUniform1i(glGetUniformLocation(self.shader_program, "tex0"), 0)
        glUniform1i(glGetUniformLocation(self.shader_program, "flipV"), 1 if flip_y else 0)

        # Render
        draw_quad_pixels(target_fbo.width, target_fbo.height)

        # Cleanup
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glUseProgram(0)
