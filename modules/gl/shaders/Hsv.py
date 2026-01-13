from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture

class Hsv(Shader):
    def use(self, tex0: Texture, hue: float, saturation: float, value: float) -> None:
        if not self.allocated or not self.shader_program:
            print("Hsv shader not allocated or shader program missing.")
            return
        if not tex0.allocated:
            print("Hsv shader: input texture not allocated.")
            return

        # Activate shader program
        glUseProgram(self.shader_program)

        # Bind input texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex0.tex_id)

        # Configure shader uniforms
        glUniform1i(glGetUniformLocation(self.shader_program, "tex0"), 0)
        glUniform1f(glGetUniformLocation(self.shader_program, "hue"), hue)
        glUniform1f(glGetUniformLocation(self.shader_program, "saturation"), saturation)
        glUniform1f(glGetUniformLocation(self.shader_program, "value"), value)

        # Render
        draw_quad()

        # Cleanup
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glUseProgram(0)

