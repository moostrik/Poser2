from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture

class Exposure(Shader):
    def use(self, tex0: Texture, exposure: float, offset: float, gamma: float) -> None:
        if not self.allocated or not self.shader_program:
            print("Exposure shader not allocated or shader program missing.")
            return
        if not tex0.allocated:
            print("Exposure shader: input texture not allocated.")
            return

        # Activate shader program
        glUseProgram(self.shader_program)

        # Bind input texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex0.tex_id)

        # Configure shader uniforms (using cached locations)
        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform1f(self.get_uniform_loc("exposure"), exposure)
        glUniform1f(self.get_uniform_loc("offset"), offset)
        glUniform1f(self.get_uniform_loc("gamma"), gamma)

        # Render
        draw_quad()

