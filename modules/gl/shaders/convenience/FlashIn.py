from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture

class FlashIn(Shader):
    def use(self, tex0: Texture, tex1: Texture, flash: float) -> None:
        if not self.allocated or not self.shader_program:
            print("FlashIn shader not allocated or shader program missing.")
            return
        if not tex0.allocated or not tex1.allocated:
            print("FlashIn shader: input textures not allocated.")
            return

        # Activate shader program
        glUseProgram(self.shader_program)

        # Bind input textures
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex0.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, tex1.tex_id)

        # Configure shader uniforms (using cached locations)
        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform1i(self.get_uniform_loc("tex1"), 1)
        glUniform1f(self.get_uniform_loc("flash"), flash)

        # Render
        draw_quad()

