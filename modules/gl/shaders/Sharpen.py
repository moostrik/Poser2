from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture

class Sharpen(Shader):
    def use(self, tex0: Texture, amount: float) -> None:
        if not self.allocated or not self.shader_program:
            print("Sharpen shader not allocated or shader program missing.")
            return
        if not tex0.allocated:
            print("Sharpen shader: input texture not allocated.")
            return

        # Activate shader program
        glUseProgram(self.shader_program)

        # Bind input texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex0.tex_id)

        # Configure shader uniforms (using cached locations)
        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform1f(self.get_uniform_loc("amount"), amount)
        glUniform2f(self.get_uniform_loc("texelSize"), 1.0 / tex0.width, 1.0 / tex0.height)

        # Render
        draw_quad()
