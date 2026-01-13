"""Velocity Direction Map shader.

Visualizes velocity field using HSV color encoding where direction maps to hue.
Ported from ofxFlowTools ftVelocityFieldShader.h
"""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture


class VelocityDirectionMap(Shader):
    """Visualize velocity field using HSV color encoding.

    Direction -> Hue, Magnitude -> Saturation/Value
    """

    def use(self, velocity_tex: Texture, scale: float = 1.0) -> None:
        """Render velocity visualization to FBO.

        Args:
            target_fbo: Target framebuffer (use screen FBO for direct rendering)
            velocity_tex: Velocity texture (RG = XY velocity)
            scale: Velocity scale multiplier
        """
        if not self.allocated or not self.shader_program:
            print("VelocityDirectionMap shader not allocated or shader program missing.")
            return
        if not velocity_tex.allocated:
            print("VelocityDirectionMap shader: input texture not allocated.")
            return

        # Activate shader program
        glUseProgram(self.shader_program)

        # Bind input texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, velocity_tex.tex_id)

        # Configure shader uniforms
        glUniform1i(glGetUniformLocation(self.shader_program, "tex0"), 0)
        glUniform1f(glGetUniformLocation(self.shader_program, "scale"), scale)

        # Render
        draw_quad()

        # Cleanup
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glUseProgram(0)
