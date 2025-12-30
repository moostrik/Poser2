from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad

class FlashOut(Shader):
    def use(self, fbo, tex0, tex1, tex2, flash: float, width: float, height: float) -> None :
        if not self.allocated: return
        if not fbo or not tex0 or not tex1: return

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex0)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, tex1)
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, tex2)

        s = self.shader_program
        glUseProgram(s)
        glUniform1i(glGetUniformLocation(s, "tex0"), 0)
        glUniform1i(glGetUniformLocation(s, "tex1"), 1)
        glUniform1i(glGetUniformLocation(s, "tex2"), 2)
        glUniform1f(glGetUniformLocation(s, "flash"), flash)
        glUniform2f(glGetUniformLocation(s, "resolution"), width, height)

        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        draw_quad()
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        glUseProgram(0)

        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)

