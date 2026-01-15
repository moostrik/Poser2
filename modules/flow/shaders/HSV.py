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
            return
        if not source.allocated:
            return

        glUseProgram(self.shader_program)

        # Bind texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, source.tex_id)

        # Set uniforms
        glUniform1i(glGetUniformLocation(self.shader_program, "tex0"), 0)
        glUniform1f(glGetUniformLocation(self.shader_program, "hue"), hue)
        glUniform1f(glGetUniformLocation(self.shader_program, "saturation"), saturation)
        glUniform1f(glGetUniformLocation(self.shader_program, "value"), value)

        draw_quad()

        # Cleanup
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glUseProgram(0)
