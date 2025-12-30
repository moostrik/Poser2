from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad
from random import random

class Noise(Shader):
    def use(self, fbo, width: float, height: float) -> None :
        if not self.allocated: return
        if not fbo: return

        s = self.shader_program
        glUseProgram(s)
        glUniform1f(glGetUniformLocation(s, "random"), random())
        glUniform2f(glGetUniformLocation(s, "resolution"), width, height)

        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        draw_quad()
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        glUseProgram(0)

