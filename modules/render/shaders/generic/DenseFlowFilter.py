from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture

class DenseFlowFilter(Shader):
    def use(self, tex0: Texture, scale: float = 1.0, gamma: float = 1.0, noise_threshold: float = 0.01) -> None:
        if not self.allocated or not self.shader_program:
            print("DenseFlowFilter shader not allocated or shader program missing.")
            return
        if not tex0.allocated:
            print("DenseFlowFilter shader: input texture not allocated.")
            return

        # Activate shader program
        glUseProgram(self.shader_program)

        # Bind input texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex0.tex_id)

        # Configure shader uniforms
        glUniform1i(glGetUniformLocation(self.shader_program, "tex0"), 0)
        glUniform1f(glGetUniformLocation(self.shader_program, "scale"), scale)
        glUniform1f(glGetUniformLocation(self.shader_program, "gamma"), gamma)
        glUniform1f(glGetUniformLocation(self.shader_program, "noiseThreshold"), noise_threshold)

        # Render
        draw_quad()

        # Cleanup
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glUseProgram(0)

