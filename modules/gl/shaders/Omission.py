from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture

class Omission(Shader):
    def use(self, tex0: Texture, tex1: Texture, mask: Texture, bloom: float, omission: float) -> None:
        if not self.allocated or not self.shader_program:
            print("Omission shader not allocated or shader program missing.")
            return
        if not tex0.allocated or not tex1.allocated or not mask.allocated:
            print("Omission shader: input textures not allocated.")
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

        # Configure shader uniforms
        glUniform1i(glGetUniformLocation(self.shader_program, "src"), 0)
        glUniform1i(glGetUniformLocation(self.shader_program, "dst"), 1)
        glUniform1i(glGetUniformLocation(self.shader_program, "mask"), 2)
        glUniform1f(glGetUniformLocation(self.shader_program, "bloom"), bloom)
        glUniform1f(glGetUniformLocation(self.shader_program, "omission"), omission)

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

