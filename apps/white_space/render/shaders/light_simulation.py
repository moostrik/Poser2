from OpenGL.GL import GL_FRAMEBUFFER, glBindFramebuffer,glUniform1f, glUniform2f, glUseProgram, glViewport
from modules.gl import Shader, draw_quad, Fbo, Texture

class LightSimulation(Shader):
    def use(self, fbo: Fbo, tex0: Texture, width: float) -> None:
        if not self.allocated or not self.shader_program: return
        if not fbo.allocated or not tex0.allocated: return

        # Activate shader program
        glUseProgram(self.shader_program)

        # Set up render target
        glBindFramebuffer(GL_FRAMEBUFFER, fbo.fbo_id)
        glViewport(0, 0, fbo.width, fbo.height)

        # Bind input texture
        tex0.bind(0)

        # Configure shader uniforms
        glUniform2f(self.get_uniform_loc("resolution"), float(fbo.width), float(fbo.height))
        glUniform1f(self.get_uniform_loc("texWidth"), width)

        # Render
        draw_quad()
