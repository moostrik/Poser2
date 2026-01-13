from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Fbo, Texture

class HDTTripleBlend(Shader):
    def use(self, fbo: Fbo,
            tex0: Texture, tex1: Texture, tex2: Texture,
            mask0: Texture, mask1: Texture, mask2: Texture,
            blend0: float, blend1: float, blend2: float,
            c1, c2, c3) -> None:
        if not self.allocated or not self.shader_program: return
        if not fbo.allocated or not tex0.allocated or not tex1.allocated or not tex2.allocated: return
        if not mask0.allocated or not mask1.allocated or not mask2.allocated: return

        # Activate shader program
        glUseProgram(self.shader_program)

        # Set up render target
        glBindFramebuffer(GL_FRAMEBUFFER, fbo.fbo_id)
        glViewport(0, 0, fbo.width, fbo.height)

        # Bind input textures
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex0.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, tex1.tex_id)
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, tex2.tex_id)
        glActiveTexture(GL_TEXTURE3)
        glBindTexture(GL_TEXTURE_2D, mask0.tex_id)
        glActiveTexture(GL_TEXTURE4)
        glBindTexture(GL_TEXTURE_2D, mask1.tex_id)
        glActiveTexture(GL_TEXTURE5)
        glBindTexture(GL_TEXTURE_2D, mask2.tex_id)

        # Configure shader uniforms
        glUniform2f(glGetUniformLocation(self.shader_program, "resolution"), float(fbo.width), float(fbo.height))
        glUniform1i(glGetUniformLocation(self.shader_program, "tex0"), 0)
        glUniform1i(glGetUniformLocation(self.shader_program, "tex1"), 1)
        glUniform1i(glGetUniformLocation(self.shader_program, "tex2"), 2)
        glUniform1i(glGetUniformLocation(self.shader_program, "mask0"), 3)
        glUniform1i(glGetUniformLocation(self.shader_program, "mask1"), 4)
        glUniform1i(glGetUniformLocation(self.shader_program, "mask2"), 5)
        glUniform1f(glGetUniformLocation(self.shader_program, "blend0"), blend0)
        glUniform1f(glGetUniformLocation(self.shader_program, "blend1"), blend1)
        glUniform1f(glGetUniformLocation(self.shader_program, "blend2"), blend2)
        glUniform4f(glGetUniformLocation(self.shader_program, "color0"), *c1)
        glUniform4f(glGetUniformLocation(self.shader_program, "color1"), *c2)
        glUniform4f(glGetUniformLocation(self.shader_program, "color2"), *c3)

        # Render
        draw_quad()

        # Cleanup (reverse order)
        glActiveTexture(GL_TEXTURE5)
        glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE4)
        glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE3)
        glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glUseProgram(0)
