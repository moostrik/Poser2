"""LayerComposite: up to three styled layers, added or each over the next."""

from OpenGL.GL import glUniform1f, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class LayerComposite(Shader):
    """Up to 3 styled RGBA layers with weights: added, or the first over the others (Porter–Duff over)."""

    def __init__(self) -> None:
        super().__init__()

    def use(self, textures: list[Texture], weights: list[float], layered: float = 1.0) -> None:
        """Composite pre-styled textures.

        Args:
            textures: Up to 3 pre-styled RGBA textures
            weights: Weight per texture
            layered: 0 = all additive, 1 = own (slot 0) in front
        """
        glUseProgram(self.shader_program)

        # Bind textures to slots 0-2
        for i, texture in enumerate(textures[:3]):
            texture.bind(i)

        # Weights
        for i in range(3):
            w = weights[i] if i < len(weights) else 0.0
            glUniform1f(self.get_uniform_loc(f"weights[{i}]"), w)

        glUniform1f(self.get_uniform_loc("layered"), layered)

        draw_quad()
