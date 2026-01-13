from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl.Texture import Texture

class TextureBlit(Shader):
    """Simple shader for drawing textures to screen using pixel coordinates.

    Uses generic vertex/fragment shaders for texture sampling.
    """

    def use(self, texture: Texture, x: float, y: float, w: float, h: float) -> None:
        """Draw texture to screen region.

        Args:
            texture: Texture to draw
            x: X position in pixels (top-left origin)
            y: Y position in pixels (top-left origin)
            w: Width in pixels
            h: Height in pixels
        """
        if not self.allocated or not self.shader_program: return
        if not texture.allocated: return

        # Activate shader program
        glUseProgram(self.shader_program)

        # Bind to screen framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(int(x), int(y), int(w), int(h))

        # Bind texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture.tex_id)

        # Configure shader uniforms
        glUniform2f(glGetUniformLocation(self.shader_program, "resolution"), float(w), float(h))
        glUniform1i(glGetUniformLocation(self.shader_program, "tex0"), 0)

        # Render quad in pixel space
        draw_quad()

        # Cleanup
        glBindTexture(GL_TEXTURE_2D, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glUseProgram(0)
