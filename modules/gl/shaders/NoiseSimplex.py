from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad_pixels
from modules.gl import Fbo
from time import time

class NoiseSimplex(Shader):
    def use(self, fbo: Fbo, speed: float, blend: float) -> None:
        if not self.allocated or not self.shader_program: return
        if not fbo.allocated: return

        t: float = (time() * speed % (3600.0))

        # Activate shader program
        glUseProgram(self.shader_program)

        # Set up render target
        glBindFramebuffer(GL_FRAMEBUFFER, fbo.fbo_id)
        glViewport(0, 0, fbo.width, fbo.height)

        # Configure shader uniforms
        glUniform2f(glGetUniformLocation(self.shader_program, "resolution"), float(fbo.width), float(fbo.height))
        glUniform1f(glGetUniformLocation(self.shader_program, "time"), t)
        glUniform1f(glGetUniformLocation(self.shader_program, "blend"), blend)

        # Render
        draw_quad_pixels(fbo.width, fbo.height)

        # Cleanup
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glUseProgram(0)

