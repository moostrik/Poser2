from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad

class MaskDilate(Shader):
    def use(self, fbo, tex0, radius: float = 1.0, texel_size: tuple | None = None) -> None:
        if not self.allocated: return
        if not fbo or not tex0: return

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex0)
        if texel_size is None:
            width = glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH)
            height = glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT)
            texel_size = (1.0 / width, 1.0 / height)

        # Bind FBO first to query its dimensions
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        s = self.shader_program
        glUseProgram(s)
        glUniform1i(glGetUniformLocation(s, "tex0"), 0)
        glUniform1f(glGetUniformLocation(s, "radius"), radius)
        glUniform2f(glGetUniformLocation(s, "texelSize"), texel_size[0], texel_size[1])
        draw_quad()
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        glUseProgram(0)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)

