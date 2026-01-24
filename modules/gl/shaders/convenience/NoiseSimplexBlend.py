from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture

class NoiseSimplexBlend(Shader):
    def use(self, tex0: Texture, tex1: Texture, mask: Texture, blend: float) -> None:
        if not self.allocated or not self.shader_program:
            print("NoiseSimplexBlend shader not allocated or shader program missing.")
            return
        if not tex0.allocated or not tex1.allocated or not mask.allocated:
            print("NoiseSimplexBlend shader: input textures not allocated.")
            return

        # Activate shader program
        glUseProgram(self.shader_program)

        # Bind input textures
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex0.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, tex1.tex_id)
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, mask.tex_id)

        # Configure shader uniforms (using cached locations)
        glUniform1i(self.get_uniform_loc("src"), 0)
        glUniform1i(self.get_uniform_loc("dst"), 1)
        glUniform1i(self.get_uniform_loc("mask"), 2)
        glUniform1f(self.get_uniform_loc("blend"), blend)

        # Render
        draw_quad()

        # Cleanup
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glUseProgram(0)

