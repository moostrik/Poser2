"""AddBoolean shader - Boolean union of obstacle masks."""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture


class AddBoolean(Shader):
    """Boolean union (OR) operation for obstacle masks."""

    def __init__(self) -> None:
        super().__init__()

    def use(self, base: Texture, blend: Texture) -> None:
        """Combine two obstacle masks with boolean OR.

        Args:
            base: Base obstacle mask
            blend: Blend obstacle mask (will be rounded to 0 or 1)

        Output:
            Union of both masks (1.0 if either input is 1.0)
        """
        # Bind shader program
        glUseProgram(self.shader_program)

        # Bind textures
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, base.tex_id)
        glUniform1i(glGetUniformLocation(self.shader_program, "uBase"), 0)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, blend.tex_id)
        glUniform1i(glGetUniformLocation(self.shader_program, "uBlend"), 1)

        # Draw fullscreen quad
        draw_quad()

        # Cleanup
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glUseProgram(0)
