"""Velocity Direction Map shader.

Visualizes velocity field using HSV color encoding where direction maps to hue.
Ported from ofxFlowTools ftVelocityFieldShader.h
"""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Fbo, Texture


class VelocityDirectionMap(Shader):
    """Visualize velocity field using HSV color encoding.

    Direction -> Hue, Magnitude -> Saturation/Value
    """

    def use(self, target_fbo: Fbo, velocity_tex: Texture, scale: float = 1.0) -> None:
        """Render velocity visualization to FBO.

        Args:
            target_fbo: Target framebuffer (use screen FBO for direct rendering)
            velocity_tex: Velocity texture (RG = XY velocity)
            scale: Velocity scale multiplier
        """
        if not self.allocated or not self.shader_program: return
        if not target_fbo.allocated or not velocity_tex.allocated: return

        # Activate shader program
        glUseProgram(self.shader_program)

        # Set up render target
        glBindFramebuffer(GL_FRAMEBUFFER, target_fbo.fbo_id)
        glViewport(0, 0, target_fbo.width, target_fbo.height)

        # Bind input texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, velocity_tex.tex_id)

        # Configure shader uniforms
        glUniform2f(glGetUniformLocation(self.shader_program, "resolution"), float(target_fbo.width), float(target_fbo.height))
        glUniform1i(glGetUniformLocation(self.shader_program, "tex0"), 0)
        glUniform1f(glGetUniformLocation(self.shader_program, "scale"), scale)

        # Render
        draw_quad()

        # Cleanup
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glUseProgram(0)
