"""Multiply Force shader.

Simple scalar multiplication for timestep scaling.
Ported from ofxFlowTools ftMultiplyForceShader.h
"""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture


class MultiplyForce(Shader):
    """Multiply texture by scalar force value."""

    def use(self, source: Texture, force: float) -> None:
        """Multiply source texture by force scalar.

        Args:
            source: Source texture
            force: Multiplier value
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
        glUniform1f(glGetUniformLocation(self.shader_program, "force"), force)

        draw_quad()

        # Cleanup
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glUseProgram(0)
