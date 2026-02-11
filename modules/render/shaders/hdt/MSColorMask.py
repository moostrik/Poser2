"""MSColorMask shader - Blend multiple masks with colors and weights."""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture


class MSColorMask(Shader):
    """Blend multiple mask textures with per-mask colors and weights.

    Takes up to 3 mask textures (R16F), colors, and weights.
    Outputs non-premultiplied RGBA where:
    - RGB = weighted color blend
    - A = combined mask alpha
    """

    def __init__(self) -> None:
        super().__init__()

    def use(
        self,
        masks: list[Texture],
        colors: list[tuple[float, float, float, float]],
        weights: list[float]
    ) -> None:
        """Colorize and blend masks with weights.

        Args:
            masks: List of up to 3 mask textures (R16F single channel)
            colors: List of up to 3 RGBA colors, one per mask
            weights: List of up to 3 weights, one per mask
        """
        if not self.allocated or not self.shader_program:
            return

        glUseProgram(self.shader_program)

        # Bind mask textures (pad with first mask if fewer than 3)
        texture_units = [GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE2]
        for i in range(3):
            glActiveTexture(texture_units[i])
            if i < len(masks) and masks[i].allocated:
                glBindTexture(GL_TEXTURE_2D, masks[i].tex_id)
            elif len(masks) > 0 and masks[0].allocated:
                glBindTexture(GL_TEXTURE_2D, masks[0].tex_id)
            glUniform1i(self.get_uniform_loc(f"uMask{i}"), i)

        # Set color uniforms (pad to 3 if fewer provided)
        padded_colors = list(colors) + [(0.0, 0.0, 0.0, 0.0)] * (3 - len(colors))
        for i, color in enumerate(padded_colors[:3]):
            glUniform4f(self.get_uniform_loc(f"uColors[{i}]"), *color)

        # Set weight uniforms (pad to 3 if fewer provided)
        padded_weights = list(weights) + [0.0] * (3 - len(weights))
        for i, weight in enumerate(padded_weights[:3]):
            glUniform1f(self.get_uniform_loc(f"uWeights[{i}]"), weight)

        draw_quad()
