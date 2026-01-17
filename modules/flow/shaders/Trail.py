"""Trail shader.

Blends current input with previous frame for temporal smoothing.
Ported from ofxFlowTools ftBridgeShader.h
"""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture


class Trail(Shader):
    """Temporal smoothing shader for any field.

    Blends previous trail with new input, with optional scaling on new input.
    """

    def use(self, prev_trail: Texture, new_input: Texture, trail_weight: float, new_weight: float = 1.0) -> None:
        """Blend previous trail with new input.

        Args:
            prev_trail: Previous trail texture
            new_input: New input texture
            trail_weight: Trail weight (0.0=replace, 0.99=keep trail)
            new_weight: New input scale/weight (default 1.0)
        """
        if not self.allocated or not self.shader_program:
            return
        if not prev_trail.allocated or not new_input.allocated:
            return

        glUseProgram(self.shader_program)

        # Bind textures
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, prev_trail.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, new_input.tex_id)

        # Set uniforms
        glUniform1i(glGetUniformLocation(self.shader_program, "tex0"), 0)
        glUniform1i(glGetUniformLocation(self.shader_program, "tex1"), 1)
        glUniform1f(glGetUniformLocation(self.shader_program, "trailWeight"), trail_weight)
        glUniform1f(glGetUniformLocation(self.shader_program, "newWeight"), new_weight)

        draw_quad()

        # Cleanup
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glUseProgram(0)
