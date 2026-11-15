from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class DepthView(Shader):
    def use(self, depth: Texture, image: Texture, near: float, far: float, mode: int) -> None:
        """Depth in metres as a picture, `mode` one of `DepthDrawMode`; `image` is drawn under it
        in OVER_IMAGE. Waits for both textures: the depth can arrive before the first image."""
        if not self.allocated or not self.shader_program or not depth.allocated or not image.allocated:
            return

        # Activate shader program
        glUseProgram(self.shader_program)

        # Bind input textures
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, depth.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, image.tex_id)

        # Configure shader uniforms (using cached locations)
        glUniform1i(self.get_uniform_loc("depth"), 0)
        glUniform1i(self.get_uniform_loc("image"), 1)
        glUniform1f(self.get_uniform_loc("nearDepth"), near)
        glUniform1f(self.get_uniform_loc("farDepth"), far)
        glUniform1i(self.get_uniform_loc("mode"), mode)

        # Render
        draw_quad()
