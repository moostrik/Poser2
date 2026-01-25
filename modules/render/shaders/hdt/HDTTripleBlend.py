from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture

class HDTTripleBlend(Shader):
    def use(self,
            tex0: Texture, tex1: Texture, tex2: Texture,
            mask0: Texture, mask1: Texture, mask2: Texture,
            blend0: float, blend1: float, blend2: float,
            c1, c2, c3) -> None:
        if not self.allocated or not self.shader_program:
            print("HDTTripleBlend shader not allocated or shader program missing.")
            return
        if not tex0.allocated or not tex1.allocated or not tex2.allocated:
            print("HDTTripleBlend shader: one or more input textures not allocated.")
            return
        if not mask0.allocated or not mask1.allocated or not mask2.allocated:
            print("HDTTripleBlend shader: one or more mask textures not allocated.")
            return

        # Activate shader program
        glUseProgram(self.shader_program)

        # Bind input textures
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex0.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, tex1.tex_id)
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, tex2.tex_id)
        glActiveTexture(GL_TEXTURE3)
        glBindTexture(GL_TEXTURE_2D, mask0.tex_id)
        glActiveTexture(GL_TEXTURE4)
        glBindTexture(GL_TEXTURE_2D, mask1.tex_id)
        glActiveTexture(GL_TEXTURE5)
        glBindTexture(GL_TEXTURE_2D, mask2.tex_id)

        # Configure shader uniforms (using cached locations)
        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform1i(self.get_uniform_loc("tex1"), 1)
        glUniform1i(self.get_uniform_loc("tex2"), 2)
        glUniform1i(self.get_uniform_loc("mask0"), 3)
        glUniform1i(self.get_uniform_loc("mask1"), 4)
        glUniform1i(self.get_uniform_loc("mask2"), 5)
        glUniform1f(self.get_uniform_loc("blend0"), blend0)
        glUniform1f(self.get_uniform_loc("blend1"), blend1)
        glUniform1f(self.get_uniform_loc("blend2"), blend2)
        glUniform4f(self.get_uniform_loc("color0"), *c1)
        glUniform4f(self.get_uniform_loc("color1"), *c2)
        glUniform4f(self.get_uniform_loc("color2"), *c3)

        # Render
        draw_quad()
