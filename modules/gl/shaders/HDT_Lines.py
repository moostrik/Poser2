from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad
from time import time

class HDT_Lines(Shader):
    def __init__(self) -> None:
        super().__init__()
        self.shader_name = self.__class__.__name__
        self.start_time = time()

    def allocate(self, monitor_file = False) -> None:
        super().allocate(self.shader_name, monitor_file)

    def use(self, fbo: int, speed: float, phase: float, anchor: float, amount: float, thickness: float, sharpness: float) -> None:
        super().use()
        if not self.allocated: return

        t = time() - self.start_time

        s = self.shader_program
        c_sharpness: float = max(min(sharpness, 0.0), 0.9999)
        
        glUseProgram(s)
        # print(time_value)
        glUniform1f(glGetUniformLocation(s, "time"), t)
        glUniform1f(glGetUniformLocation(s, "speed"), speed)
        glUniform1f(glGetUniformLocation(s, "phase"), phase)
        glUniform1f(glGetUniformLocation(s, "anchor"), anchor)
        glUniform1f(glGetUniformLocation(s, "amount"), amount)
        glUniform1f(glGetUniformLocation(s, "thickness"), thickness)
        glUniform1f(glGetUniformLocation(s, "sharpness"), c_sharpness)

        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        draw_quad()
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        glUseProgram(0)

        glBindTexture(GL_TEXTURE_2D, 0)

