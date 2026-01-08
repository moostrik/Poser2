"""Add Multiplied shader.

Adds two textures with individual multipliers.
Ported from ofxFlowTools ftAddMultipliedShader.h
"""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Fbo, Texture


class AddMultiplied(Shader):
    """Add two textures with individual strength multipliers."""

    def use(self, target_fbo: Fbo, tex0: Texture, tex1: Texture,
             strength0: float = 1.0, strength1: float = 1.0) -> None:
        """Render blended result to FBO.

        Args:
            target_fbo: Target framebuffer
            tex0: First source texture
            tex1: Second source texture
            strength0: Multiplier for first texture
            strength1: Multiplier for second texture
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
        glBindTexture(GL_TEXTURE_2D, tex0.tex_id)
        glUniform1i(glGetUniformLocation(self.shader_program, "tex0"), 0)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, tex1.tex_id)
        glUniform1i(glGetUniformLocation(self.shader_program, "tex1"), 1)

        glUniform1f(glGetUniformLocation(self.shader_program, "strength0"), strength0)
        glUniform1f(glGetUniformLocation(self.shader_program, "strength1"), strength1)

        draw_quad()

        glEnable(GL_BLEND)
        glUseProgram(0)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
