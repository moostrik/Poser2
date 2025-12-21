from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad

class FlowFilter(Shader):
    def __init__(self) -> None:
        super().__init__()
        self.shader_name = self.__class__.__name__
        self.scale: float = 1.0
        self.gamma: float = 1.0
        self.noise_threshold: float = 0.01

    def allocate(self, monitor_file = False) -> None:
        super().allocate(self.shader_name, monitor_file)

    def use(self, fbo, tex0, scale: float = 1.0, gamma: float = 1.0, noise_threshold: float = 0.01) -> None:
        super().use()
        if not self.allocated: return
        if not fbo or not tex0: return

        self.scale = scale
        self.gamma = gamma
        self.noise_threshold = noise_threshold

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex0)

        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        s = self.shader_program
        glUseProgram(s)
        glUniform1i(glGetUniformLocation(s, "tex0"), 0)
        glUniform1f(glGetUniformLocation(s, "scale"), self.scale)
        glUniform1f(glGetUniformLocation(s, "gamma"), self.gamma)
        glUniform1f(glGetUniformLocation(s, "noiseThreshold"), self.noise_threshold)
        draw_quad()
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        glUseProgram(0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)

