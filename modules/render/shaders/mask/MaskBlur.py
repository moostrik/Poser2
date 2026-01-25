from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture

class MaskBlur(Shader):
    def use(self, tex0: Texture, horizontal: bool = True, radius: float = 1.0, texel_size: tuple[float, float] | None = None) -> None:
        if not self.allocated or not self.shader_program:
            print("MaskBlur shader not allocated or shader program missing.")
            return
        if not tex0.allocated:
            print("MaskBlur shader: input texture not allocated.")
            return

        if texel_size is None:
            texel_size = (1.0 / tex0.width, 1.0 / tex0.height)

        # Set direction based on horizontal flag
        direction = (1.0, 0.0) if horizontal else (0.0, 1.0)

        # Activate shader program
        glUseProgram(self.shader_program)

        # Bind input texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex0.tex_id)

        # Configure shader uniforms (using cached locations)
        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform2f(self.get_uniform_loc("direction"), direction[0], direction[1])
        glUniform2f(self.get_uniform_loc("texelSize"), texel_size[0], texel_size[1])
        glUniform1f(self.get_uniform_loc("blurRadius"), radius)

        # Render
        draw_quad()
