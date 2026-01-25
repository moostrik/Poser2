from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture

class MaskApply(Shader):
    def use(self, color: Texture, mask: Texture, multiply: float = 1.0) -> None:
        if not self.allocated or not self.shader_program:
            print("MaskApply shader not allocated or shader program missing.")
            return
        if not color.allocated or not mask.allocated:
            print("MaskApply shader: input textures not allocated.")
            return

        # Activate shader program
        glUseProgram(self.shader_program)

        # Bind input textures
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, color.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, mask.tex_id)

        # Configure shader uniforms (using cached locations)
        glUniform1i(self.get_uniform_loc("color"), 0)
        glUniform1i(self.get_uniform_loc("mask"), 1)
        glUniform1f(self.get_uniform_loc("multiply"), multiply)

        # Render
        draw_quad()

