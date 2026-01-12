from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad_pixels
from modules.gl import Fbo

class HDT_Lines(Shader):
    def use(self, fbo: Fbo, time: float, phase: float, anchor: float, amount: float, thickness: float, sharpness: float, stretch: float, mess: float,
            param01: float = 0.0, param02: float = 0.0, param03: float = 0.0, param04: float = 0.0, param05: float = 0.0) -> None:
        if not self.allocated or not self.shader_program: return
        if not fbo.allocated: return

        c_sharpness: float = min(max(sharpness, 0.0), 0.9999)

        # Activate shader program
        glUseProgram(self.shader_program)

        # Set up render target
        glBindFramebuffer(GL_FRAMEBUFFER, fbo.fbo_id)
        glViewport(0, 0, fbo.width, fbo.height)

        # Configure shader uniforms
        glUniform2f(glGetUniformLocation(self.shader_program, "resolution"), float(fbo.width), float(fbo.height))
        glUniform1f(glGetUniformLocation(self.shader_program, "time"), time)
        glUniform1f(glGetUniformLocation(self.shader_program, "speed"), 1.0)
        glUniform1f(glGetUniformLocation(self.shader_program, "phase"), phase)
        glUniform1f(glGetUniformLocation(self.shader_program, "anchor"), anchor)
        glUniform1f(glGetUniformLocation(self.shader_program, "amount"), amount)
        glUniform1f(glGetUniformLocation(self.shader_program, "thickness"), thickness)
        glUniform1f(glGetUniformLocation(self.shader_program, "sharpness"), c_sharpness)
        glUniform1f(glGetUniformLocation(self.shader_program, "stretch"), stretch)
        glUniform1f(glGetUniformLocation(self.shader_program, "mess"), mess)
        glUniform1f(glGetUniformLocation(self.shader_program, "param01"), param01)
        glUniform1f(glGetUniformLocation(self.shader_program, "param02"), param02)
        glUniform1f(glGetUniformLocation(self.shader_program, "param03"), param03)
        glUniform1f(glGetUniformLocation(self.shader_program, "param04"), param04)
        glUniform1f(glGetUniformLocation(self.shader_program, "param05"), param05)

        # Render
        draw_quad_pixels(fbo.width, fbo.height)

        # Cleanup
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glUseProgram(0)
