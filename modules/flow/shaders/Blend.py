"""Add Multiplied shader.

Adds two textures with individual multipliers.
Ported from ofxFlowTools ftAddMultipliedShader.h
"""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture


class Blend(Shader):
    """Add two textures with individual strength multipliers."""

    def use(self, dst: Texture, src: Texture,
             dst_strength: float = 1.0, src_strength: float = 1.0) -> None:
        """Render blended result to FBO.

        Args:
            target_fbo: Target framebuffer
            tex0: First source texture
            tex1: Second source texture
            strength0: Multiplier for first texture
            strength1: Multiplier for second texture
        """
        if not self.allocated or not self.shader_program:
            print("Blend shader not allocated or shader program missing.")
            return
        if not dst.allocated or not src.allocated:
            print("Blend shader: input textures not allocated.")
            return

        # Activate shader program
        glUseProgram(self.shader_program)

        # Bind input textures
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, dst.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, src.tex_id)

        # Configure shader uniforms
        glUniform1i(glGetUniformLocation(self.shader_program, "dst"), 0)
        glUniform1i(glGetUniformLocation(self.shader_program, "src"), 1)
        glUniform1f(glGetUniformLocation(self.shader_program, "dst_strength"), dst_strength)
        glUniform1f(glGetUniformLocation(self.shader_program, "src_strength"), src_strength)

        # Render
        draw_quad()

        # Cleanup
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glUseProgram(0)
