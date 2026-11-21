"""AddBoolean shader - Boolean union of obstacle masks."""

from OpenGL.GL import glUseProgram
from modules.gl import Shader, draw_quad, Texture


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
        base.bind(0)
        blend.bind(1)

        # Draw fullscreen quad
        draw_quad()
