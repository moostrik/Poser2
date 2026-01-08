from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad

class HDTTripleBlend(Shader):
    def use(self, fbo,
            tex0, tex1, tex2,
            mask0, mask1, mask2,
            blend0: float, blend1: float, blend2: float,
            c1, c2, c3) -> None :
        if not self.allocated: return
        if not fbo or not tex0 or not tex1: return

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex0)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, tex1)
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, tex2)
        glActiveTexture(GL_TEXTURE3)
        glBindTexture(GL_TEXTURE_2D, mask0)
        glActiveTexture(GL_TEXTURE4)
        glBindTexture(GL_TEXTURE_2D, mask1)
        glActiveTexture(GL_TEXTURE5)
        glBindTexture(GL_TEXTURE_2D, mask2)

        s = self.shader_program
        glUseProgram(s)
        glUniform1i(glGetUniformLocation(s, "tex0"), 0)
        glUniform1i(glGetUniformLocation(s, "tex1"), 1)
        glUniform1i(glGetUniformLocation(s, "tex2"), 2)
        glUniform1i(glGetUniformLocation(s, "mask0"), 3)
        glUniform1i(glGetUniformLocation(s, "mask1"), 4)
        glUniform1i(glGetUniformLocation(s, "mask2"), 5)
        glUniform1f(glGetUniformLocation(s, "blend0"), blend0)
        glUniform1f(glGetUniformLocation(s, "blend1"), blend1)
        glUniform1f(glGetUniformLocation(s, "blend2"), blend2)
        glUniform4f(glGetUniformLocation(s, "color0"), *c1)
        glUniform4f(glGetUniformLocation(s, "color1"), *c2)
        glUniform4f(glGetUniformLocation(s, "color2"), *c3)

        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        draw_quad()
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        glUseProgram(0)

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
