from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad

class Exposure(Shader):
    def use(self, fbo, tex0, exposure: float, offset: float, gamma: float) -> None :
        if not self.allocated: return
        if not fbo or not tex0: return

        glBindTexture(GL_TEXTURE_2D, tex0)

        s = self.shader_program
        glUseProgram(s)
        glUniform1i(glGetUniformLocation(s, "tex0"), 0)
        glUniform1f(glGetUniformLocation(s, "exposure"), exposure)
        glUniform1f(glGetUniformLocation(s, "offset"), offset)
        glUniform1f(glGetUniformLocation(s, "gamma"), gamma)

        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        draw_quad()
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        glUseProgram(0)

        glBindTexture(GL_TEXTURE_2D, 0)

