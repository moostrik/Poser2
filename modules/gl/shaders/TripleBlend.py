from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad

class TripleBlend(Shader):
    def __init__(self) -> None:
        super().__init__()
        self.shader_name = self.__class__.__name__

    def allocate(self, monitor_file = False) -> None:
        super().allocate(self.shader_name, monitor_file)

    def use(self, fbo, tex0, tex1, tex2, blend1: float, blend2: float) -> None :
        super().use()
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
        glUniform1f(glGetUniformLocation(s, "blend1"), blend1)
        glUniform1f(glGetUniformLocation(s, "blend2"), blend2)

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
