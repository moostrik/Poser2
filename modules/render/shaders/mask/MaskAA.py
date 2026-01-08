from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad

class MaskAA(Shader):
    def use(self, fbo, tex0, texel_size: tuple | None = None, blur_radius: float = 1.0, aa_mode: int = 1) -> None:
        if not self.allocated: return
        if not fbo or not tex0: return

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex0)
        if texel_size is None:
            width = glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH)
            height = glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT)
            texel_size = (1.0 / width, 1.0 / height)

        s = self.shader_program
        glUseProgram(s)
        glUniform1i(glGetUniformLocation(s, "tex0"), 0)
        glUniform1i(glGetUniformLocation(s, "aaMode"), aa_mode)
        glUniform1f(glGetUniformLocation(s, "blurRadius"), blur_radius)
        glUniform2f(glGetUniformLocation(s, "texelSize"), texel_size[0], texel_size[1])

        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        draw_quad()
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        glUseProgram(0)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)

