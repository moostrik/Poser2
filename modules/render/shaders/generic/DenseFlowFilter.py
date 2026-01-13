from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Fbo, Texture

class DenseFlowFilter(Shader):
    def use(self, fbo: Fbo, tex0: Texture, scale: float = 1.0, gamma: float = 1.0, noise_threshold: float = 0.01) -> None:
        if not self.allocated or not self.shader_program: return
        if not fbo.allocated or not tex0.allocated: return

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
        glUniform1f(glGetUniformLocation(self.shader_program, "scale"), scale)
        glUniform1f(glGetUniformLocation(self.shader_program, "gamma"), gamma)
        glUniform1f(glGetUniformLocation(self.shader_program, "noiseThreshold"), noise_threshold)

        # Render
        draw_quad()

        # Cleanup (reverse order)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glUseProgram(0)

