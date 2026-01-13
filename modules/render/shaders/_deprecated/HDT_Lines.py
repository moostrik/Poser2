from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad

class HDT_Lines(Shader):
    def use(self, time: float, phase: float, anchor: float, amount: float, thickness: float, sharpness: float, stretch: float, mess: float,
            param01: float = 0.0, param02: float = 0.0, param03: float = 0.0, param04: float = 0.0, param05: float = 0.0) -> None:
        if not self.allocated or not self.shader_program:
            print("HDT_Lines shader not allocated or shader program missing.")
            return

        c_sharpness: float = min(max(sharpness, 0.0), 0.9999)

        # Activate shader program
        glUseProgram(self.shader_program)

        # Configure shader uniforms
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
        draw_quad()

        # Cleanup
        glUseProgram(0)
