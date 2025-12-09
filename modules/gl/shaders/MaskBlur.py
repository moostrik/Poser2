from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad

class MaskBlur(Shader):
    def __init__(self) -> None:
        super().__init__()
        self.shader_name = self.__class__.__name__

    def allocate(self, monitor_file = False) -> None:
        super().allocate(self.shader_name, monitor_file)

    def use(self, fbo, tex0, horizontal: bool = True, radius: float = 1.0, texel_size: tuple[float, float] | None = None) -> None:
        super().use()
        if not self.allocated: return
        if not fbo or not tex0: return

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex0)
        if texel_size is None:
            width = glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH)
            height = glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT)
            texel_size = (1.0 / width, 1.0 / height)

        # Set direction based on horizontal flag
        direction = (1.0, 0.0) if horizontal else (0.0, 1.0)

        s = self.shader_program
        glUseProgram(s)
        glUniform1i(glGetUniformLocation(s, "tex0"), 0)
        glUniform2f(glGetUniformLocation(s, "direction"), direction[0], direction[1])
        glUniform2f(glGetUniformLocation(s, "texelSize"), texel_size[0], texel_size[1])
        glUniform1f(glGetUniformLocation(s, "blurRadius"), radius)

        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        draw_quad()
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        glUseProgram(0)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)
