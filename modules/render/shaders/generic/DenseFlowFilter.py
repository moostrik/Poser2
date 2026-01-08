from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad

class DenseFlowFilter(Shader):
    def use(self, fbo, tex0, scale: float = 1.0, gamma: float = 1.0, noise_threshold: float = 0.01) -> None:
        if not self.allocated: return
        if not fbo or not tex0: return

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex0)

        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        s = self.shader_program
        glUseProgram(s)
        glUniform1i(glGetUniformLocation(s, "tex0"), 0)
        glUniform1f(glGetUniformLocation(s, "scale"), scale)
        glUniform1f(glGetUniformLocation(s, "gamma"), gamma)
        glUniform1f(glGetUniformLocation(s, "noiseThreshold"), noise_threshold)
        draw_quad()
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        glUseProgram(0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)

