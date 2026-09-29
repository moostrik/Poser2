from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad
from random import random
from time import time

class NoiseSimplex(Shader):
    def __init__(self) -> None:
        super().__init__()
        self.shader_name = self.__class__.__name__

    def allocate(self, monitor_file = False) -> None:
        super().allocate(self.shader_name, monitor_file)

    def use(self, fbo, blend: float, width: float, height: float) -> None :
        super().use()
        if not self.allocated: return
        if not fbo: return

        t: float = (time() % (3600.0))

        s = self.shader_program
        glUseProgram(s)
        glUniform1f(glGetUniformLocation(s, "time"), t)
        glUniform1f(glGetUniformLocation(s, "blend"), blend)
        glUniform2f(glGetUniformLocation(s, "resolution"), width, height)

        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        draw_quad()
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        glUseProgram(0)

