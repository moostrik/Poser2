"""Multiply Force shader.

Simple scalar multiplication for timestep scaling.
Ported from ofxFlowTools ftMultiplyForceShader.h
"""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture


class Scale(Shader):
    """Multiply texture by scalar force value."""

    def use(self, src: Texture, scale: float) -> None:
        """Multiply source texture by force scalar.

        Args:
            source: Source texture
            force: Multiplier value
        """
        if not self.allocated or not self.shader_program:
            print("MultiplyForce shader not allocated or shader program missing.")
            return
        if not src.allocated:
            print("MultiplyForce shader: input texture not allocated.")
            return

        glUseProgram(self.shader_program)

        # Bind texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, src.tex_id)

        # Set uniforms (using cached locations)
        glUniform1i(self.get_uniform_loc("src"), 0)
        glUniform1f(self.get_uniform_loc("scale"), scale)

        draw_quad()
