from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture

class HDT_LineBlend(Shader):
    def use(self, tex0: Texture, line_tex: Texture, color: tuple[float, float, float, float], visibility: float, param0: float, param1: float) -> None:
        if not self.allocated or not self.shader_program:
            print("HDT_LineBlend shader not allocated or shader program missing.")
            return
        if not tex0.allocated:
            print("HDT_LineBlend shader: input texture not allocated.")
            return

        # Activate shader program
        glUseProgram(self.shader_program)

        # Bind input textures
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex0.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, line_tex.tex_id)

        # Configure shader uniforms (using cached locations)
        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform1i(self.get_uniform_loc("line_tex"), 1)
        glUniform4f(self.get_uniform_loc("target_color"), *color)
        glUniform1f(self.get_uniform_loc("visibility"), visibility)
        glUniform1f(self.get_uniform_loc("param0"), param0)
        glUniform1f(self.get_uniform_loc("param1"), param1)

        # Render
        draw_quad()
