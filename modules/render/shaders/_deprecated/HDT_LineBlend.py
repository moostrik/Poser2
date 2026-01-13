from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Fbo, Texture

class HDT_LineBlend(Shader):
    def use(self, fbo: Fbo, tex0: Texture, line_tex: Texture, color: tuple[float, float, float, float], visibility: float, param0: float, param1: float) -> None:
        if not self.allocated or not self.shader_program: return
        if not fbo.allocated or not tex0.allocated: return

        # Activate shader program
        glUseProgram(self.shader_program)

        # Set up render target
        glBindFramebuffer(GL_FRAMEBUFFER, fbo.fbo_id)
        glViewport(0, 0, fbo.width, fbo.height)

        # Bind input textures
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex0.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, line_tex.tex_id)

        # Configure shader uniforms
        glUniform2f(glGetUniformLocation(self.shader_program, "resolution"), float(fbo.width), float(fbo.height))
        glUniform1i(glGetUniformLocation(self.shader_program, "tex0"), 0)
        glUniform1i(glGetUniformLocation(self.shader_program, "line_tex"), 1)
        glUniform4f(glGetUniformLocation(self.shader_program, "target_color"), *color)
        glUniform1f(glGetUniformLocation(self.shader_program, "visibility"), visibility)
        glUniform1f(glGetUniformLocation(self.shader_program, "param0"), param0)
        glUniform1f(glGetUniformLocation(self.shader_program, "param1"), param1)

        # Render
        draw_quad()

        # Cleanup
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glUseProgram(0)
