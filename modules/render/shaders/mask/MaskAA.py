from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad_pixels
from modules.gl import Fbo, Texture

class MaskAA(Shader):
    def use(self, fbo: Fbo, tex0: Texture, texel_size: tuple | None = None, blur_radius: float = 1.0, aa_mode: int = 1) -> None:
        if not self.allocated or not self.shader_program: return
        if not fbo.allocated or not tex0.allocated: return

        if texel_size is None:
            texel_size = (1.0 / tex0.width, 1.0 / tex0.height)

        # Activate shader program
        glUseProgram(self.shader_program)

        # Set up render target
        glBindFramebuffer(GL_FRAMEBUFFER, fbo.fbo_id)
        glViewport(0, 0, fbo.width, fbo.height)

        # Bind input texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex0.tex_id)

        # Configure shader uniforms
        glUniform2f(glGetUniformLocation(self.shader_program, "resolution"), float(fbo.width), float(fbo.height))
        glUniform1i(glGetUniformLocation(self.shader_program, "tex0"), 0)
        glUniform1i(glGetUniformLocation(self.shader_program, "aaMode"), aa_mode)
        glUniform1f(glGetUniformLocation(self.shader_program, "blurRadius"), blur_radius)
        glUniform2f(glGetUniformLocation(self.shader_program, "texelSize"), texel_size[0], texel_size[1])

        # Render
        draw_quad_pixels(fbo.width, fbo.height)

        # Cleanup
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glUseProgram(0)

