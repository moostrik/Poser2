from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad

class BlendWithMask(Shader):
    def __init__(self) -> None:
        super().__init__()
        self.shader_name = self.__class__.__name__

    def allocate(self, monitor_file = False) -> None:
        super().allocate(self.shader_name, monitor_file)

    def use(self, fbo, prev, curr, fade: float, texel_size: tuple | None = None, blur_radius: float = 1.0) -> None:
        super().use()
        if not self.allocated: return
        if not fbo or not prev or not curr: return

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, prev)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, curr)

        s = self.shader_program
        glUseProgram(s)
        glUniform1i(glGetUniformLocation(s, "prev"), 0)
        glUniform1i(glGetUniformLocation(s, "curr"), 1)
        glUniform1f(glGetUniformLocation(s, "blend"), fade)
        glUniform1f(glGetUniformLocation(s, "blurRadius"), blur_radius)

        if texel_size:
            glUniform2f(glGetUniformLocation(s, "texelSize"), texel_size[0], texel_size[1])
        else:
            glUniform2f(glGetUniformLocation(s, "texelSize"), 0.001, 0.001)  # Default

        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        draw_quad()
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        glUseProgram(0)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)

