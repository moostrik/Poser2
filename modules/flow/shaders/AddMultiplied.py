"""Add Multiplied shader.

Adds two textures with individual multipliers.
Ported from ofxFlowTools ftAddMultipliedShader.h
"""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture


class AddMultiplied(Shader):
    """Add two textures with individual strength multipliers."""

    def use(self, tex0: Texture, tex1: Texture,
             strength0: float = 1.0, strength1: float = 1.0) -> None:
        """Render blended result to FBO.

        Args:
            target_fbo: Target framebuffer
            tex0: First source texture
            tex1: Second source texture
            strength0: Multiplier for first texture
            strength1: Multiplier for second texture
        """
        if not self.allocated or not self.shader_program:
            print("AddMultiplied shader not allocated or shader program missing.")
            return
        if not tex0.allocated or not tex1.allocated:
            print("AddMultiplied shader: input textures not allocated.")
            return

        # Activate shader program
        glUseProgram(self.shader_program)

        # Bind input textures
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex0.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, tex1.tex_id)

        # Configure shader uniforms
        glUniform1i(glGetUniformLocation(self.shader_program, "tex0"), 0)
        glUniform1i(glGetUniformLocation(self.shader_program, "tex1"), 1)
        glUniform1f(glGetUniformLocation(self.shader_program, "strength0"), strength0)
        glUniform1f(glGetUniformLocation(self.shader_program, "strength1"), strength1)

        # Render
        draw_quad()

        # Cleanup
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glUseProgram(0)
