from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture

class HD_Sync(Shader):
    def use(self, tex0: Texture, tex1: Texture, tex2: Texture, noise: Texture, blend1: float, blend2: float) -> None:
        if not self.allocated or not self.shader_program:
            print("HD_Sync shader not allocated or shader program missing.")
            return
        if not tex0.allocated or not tex1.allocated or not tex2.allocated:
            print("HD_Sync shader: one or more input textures not allocated.")
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
        glBindTexture(GL_TEXTURE_2D, noise.tex_id)

        # Configure shader uniforms
        glUniform1i(glGetUniformLocation(self.shader_program, "tex0"), 0)
        glUniform1i(glGetUniformLocation(self.shader_program, "tex1"), 1)
        glUniform1i(glGetUniformLocation(self.shader_program, "tex2"), 2)
        glUniform1i(glGetUniformLocation(self.shader_program, "noise"), 3)
        glUniform1f(glGetUniformLocation(self.shader_program, "blend1"), blend1)
        glUniform1f(glGetUniformLocation(self.shader_program, "blend2"), blend2)

        # Render
        draw_quad()

        # Cleanup
        glActiveTexture(GL_TEXTURE3)
        glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glUseProgram(0)

