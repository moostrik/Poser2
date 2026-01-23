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
        glUniform1f(self.get_uniform_loc("time"), time)
        glUniform1f(self.get_uniform_loc("speed"), 1.0)
        glUniform1f(self.get_uniform_loc("phase"), phase)
        glUniform1f(self.get_uniform_loc("anchor"), anchor)
        glUniform1f(self.get_uniform_loc("amount"), amount)
        glUniform1f(self.get_uniform_loc("thickness"), thickness)
        glUniform1f(self.get_uniform_loc("sharpness"), c_sharpness)
        glUniform1f(self.get_uniform_loc("stretch"), stretch)
        glUniform1f(self.get_uniform_loc("mess"), mess)
        glUniform1f(self.get_uniform_loc("param01"), param01)
        glUniform1f(self.get_uniform_loc("param02"), param02)
        glUniform1f(self.get_uniform_loc("param03"), param03)
        glUniform1f(self.get_uniform_loc("param04"), param04)
        glUniform1f(self.get_uniform_loc("param05"), param05)

        # Render
        draw_quad()

        # Cleanup
        glUseProgram(0)
