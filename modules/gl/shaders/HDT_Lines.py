from OpenGL.GL import * # type: ignore
from OpenGL.GL.shaders import ShaderProgram
from modules.gl.Shader import Shader, draw_quad

from modules.utils.HotReloadMethods import HotReloadMethods

class HDT_Lines(Shader):
    def __init__(self) -> None:
        super().__init__()
        self.shader_name = self.__class__.__name__

        # self.hot_reloader = HotReloadMethods(self.__class__, True)

    def allocate(self, monitor_file = False) -> None:
        super().allocate(self.shader_name, monitor_file)

    def use(self, fbo: int, time: float, phase: float, anchor: float, amount: float, thickness: float, sharpness: float, stretch: float, mess: float,
            param01: float = 0.0, param02: float = 0.0, param03: float = 0.0, param04: float = 0.0, param05: float = 0.0) -> None:
        super().use()
        if not self.allocated:
            return
        s: ShaderProgram | None = self.shader_program
        if s is None:
            return

        c_sharpness: float = max(min(sharpness, 0.0), 0.9999)

        glUseProgram(s)
        glUniform1f(glGetUniformLocation(s, "time"), time)
        glUniform1f(glGetUniformLocation(s, "speed"), 1.0)
        glUniform1f(glGetUniformLocation(s, "phase"), phase)
        glUniform1f(glGetUniformLocation(s, "anchor"), anchor)
        glUniform1f(glGetUniformLocation(s, "amount"), amount)
        glUniform1f(glGetUniformLocation(s, "thickness"), thickness)
        glUniform1f(glGetUniformLocation(s, "sharpness"), c_sharpness)
        glUniform1f(glGetUniformLocation(s, "stretch"), stretch)
        glUniform1f(glGetUniformLocation(s, "mess"), mess)
        glUniform1f(glGetUniformLocation(s, "param01"), param01)
        glUniform1f(glGetUniformLocation(s, "param02"), param02)
        glUniform1f(glGetUniformLocation(s, "param03"), param03)
        glUniform1f(glGetUniformLocation(s, "param04"), param04)
        glUniform1f(glGetUniformLocation(s, "param05"), param05)

        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        draw_quad()
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        glUseProgram(0)

        glBindTexture(GL_TEXTURE_2D, 0)

