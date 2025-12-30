from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad

class NoiseBlend(Shader):
    def use(self, fbo, tex0, tex1, mask, blend: float) -> None :
        if not self.allocated: return
        if not fbo or not tex0 or not tex1: return

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex0)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, tex1)
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, mask)

        s = self.shader_program
        glUseProgram(s)
        glUniform1i(glGetUniformLocation(s, "src"), 0)
        glUniform1i(glGetUniformLocation(s, "dst"), 1)
        glUniform1i(glGetUniformLocation(s, "mask"), 2)
        glUniform1f(glGetUniformLocation(s, "blend"), blend)

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

