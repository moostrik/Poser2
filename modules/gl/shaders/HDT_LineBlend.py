from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad

class HDT_LineBlend(Shader):
    def use(self, fbo, tex0, line_tex, color: tuple[float, float, float, float],visibility: float, param0: float, param1: float) -> None :
        if not self.allocated: return
        if not fbo or not tex0: return

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex0)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, line_tex)

        s = self.shader_program
        glUseProgram(s)
        glUniform1i(glGetUniformLocation(s, "tex0"), 0)
        glUniform1i(glGetUniformLocation(s, "line_tex"), 1)
        glUniform4f(glGetUniformLocation(s, "target_color"), *color)
        glUniform1f(glGetUniformLocation(s, "visibility"), visibility)
        glUniform1f(glGetUniformLocation(s, "param0"), param0)
        glUniform1f(glGetUniformLocation(s, "param1"), param1)

        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        draw_quad()
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        glUseProgram(0)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)
