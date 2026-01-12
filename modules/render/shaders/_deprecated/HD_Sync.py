from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad_pixels
from modules.gl import Fbo, Texture

class HD_Sync(Shader):
    def use(self, fbo: Fbo, tex0: Texture, tex1: Texture, tex2: Texture, noise: Texture, blend1: float, blend2: float) -> None:
        if not self.allocated or not self.shader_program: return
        if not fbo.allocated or not tex0.allocated or not tex1.allocated or not tex2.allocated: return

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
        glBindTexture(GL_TEXTURE_2D, noise.tex_id)

        # Configure shader uniforms
        glUniform2f(glGetUniformLocation(self.shader_program, "resolution"), float(fbo.width), float(fbo.height))
        glUniform1i(glGetUniformLocation(self.shader_program, "tex0"), 0)
        glUniform1i(glGetUniformLocation(self.shader_program, "tex1"), 1)
        glUniform1i(glGetUniformLocation(self.shader_program, "tex2"), 2)
        glUniform1i(glGetUniformLocation(self.shader_program, "noise"), 3)
        glUniform1f(glGetUniformLocation(self.shader_program, "blend1"), blend1)
        glUniform1f(glGetUniformLocation(self.shader_program, "blend2"), blend2)

        # Render
        draw_quad_pixels(fbo.width, fbo.height)

        # Cleanup
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

