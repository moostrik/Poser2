"""MSColorMask shader - Composite pre-styled RGBA textures."""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture


class MSColorMask(Shader):
    """Composite up to 3 pre-styled RGBA textures with weights."""

    def __init__(self) -> None:
        super().__init__()

    def use(self, textures: list[Texture], weights: list[float]) -> None:
        """Composite pre-styled textures.

        Args:
            textures: Up to 3 pre-styled RGBA textures
            weights: Weight per texture
        """
        glUseProgram(self.shader_program)

        # Bind textures to slots 0-2
        for i in range(3):
            glActiveTexture(int(GL_TEXTURE0) + i)
            if i < len(textures):
                glBindTexture(GL_TEXTURE_2D, textures[i].tex_id)
            glUniform1i(self.get_uniform_loc(f"uTex{i}"), i)

        # Weights
        for i in range(3):
            w = weights[i] if i < len(weights) else 0.0
            glUniform1f(self.get_uniform_loc(f"uWeights[{i}]"), w)

        draw_quad()
