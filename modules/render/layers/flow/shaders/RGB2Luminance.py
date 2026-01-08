"""RGB to Luminance shader.

Converts RGB image to single-channel luminance for optical flow input.
Ported from ofxFlowTools ftRGB2LuminanceShader.h
"""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Fbo, Texture


class RGB2Luminance(Shader):
    """Convert RGB texture to luminance (grayscale)."""

    def use(self, target_fbo: Fbo, source_tex: Texture) -> None:
        """Render luminance to FBO.

        Args:
            target_fbo: Target framebuffer
            source_tex: Source RGB texture
        """
        if not self.allocated:
            return
        if self.shader_program is None:
            return

        glBindFramebuffer(GL_FRAMEBUFFER, target_fbo.fbo_id)
        glViewport(0, 0, target_fbo.width, target_fbo.height)
        glDisable(GL_BLEND)

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, source_tex.tex_id)
        glUniform1i(glGetUniformLocation(self.shader_program, "tex0"), 0)

        draw_quad()

        glEnable(GL_BLEND)
        glUseProgram(0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
