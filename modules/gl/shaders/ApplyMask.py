from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad

class ApplyMask(Shader):
    def use(self, fbo, color, mask, multiply: float = 1.0) -> None :
        if not self.allocated: return
        if not fbo or not color or not mask: return

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, color)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, mask)

        s = self.shader_program
        glUseProgram(s)
        glUniform1i(glGetUniformLocation(s, "color"), 0)
        glUniform1i(glGetUniformLocation(s, "mask"), 1)
        glUniform1f(glGetUniformLocation(s, "multiply"), multiply)

        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        draw_quad()
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        glUseProgram(0)

        glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)

