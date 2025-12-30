from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad

class Hsv(Shader):
    def use(self, fbo, tex0, hue: float, saturation: float, value: float) -> None :
        if not self.allocated: return
        if not fbo or not tex0: return

        glBindTexture(GL_TEXTURE_2D, tex0)

        s = self.shader_program
        glUseProgram(s)
        glUniform1i(glGetUniformLocation(s, "tex0"), 0)
        glUniform1f(glGetUniformLocation(s, "hue"), hue)
        glUniform1f(glGetUniformLocation(s, "saturation"), saturation)
        glUniform1f(glGetUniformLocation(s, "value"), value)

        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        draw_quad()
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        glUseProgram(0)

        glBindTexture(GL_TEXTURE_2D, 0)

