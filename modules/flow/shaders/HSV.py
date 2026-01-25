"""HSV Color Adjustment shader.

Adjusts hue, saturation, and value of RGB colors.
Ported from ofxFlowTools ftHSVShader.h
"""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture


class HSV(Shader):
    """HSV color adjustment shader."""

    def use(self, source: Texture, hue: float = 0.0,
            saturation: float = 1.0, value: float = 1.0) -> None:
        """Apply HSV adjustments to source texture.

        Args:
            source: Source texture (RGB or RGBA)
            hue: Hue shift in range [-0.5, 0.5] (0 = no change)
            saturation: Saturation multiplier (0 = grayscale, 1 = original, >1 = boosted)
            value: Value/brightness multiplier (0 = black, 1 = original, >1 = brighter)
        """
        if not self.allocated or not self.shader_program:
            print("HSV shader not allocated or shader program missing.")
            return
        if not source.allocated:
            print("HSV shader: input texture not allocated.")
            return

        glUseProgram(self.shader_program)

        # Bind texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, source.tex_id)

        # Set uniforms (using cached locations)
        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform1f(self.get_uniform_loc("hue"), hue)
        glUniform1f(self.get_uniform_loc("saturation"), saturation)
        glUniform1f(self.get_uniform_loc("value"), value)

        draw_quad()
