"""Velocity Arrow Field shader.

Visualizes velocity field using procedural arrows without geometry shaders.
Ported from ofxFlowTools visualization concepts.
"""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Fbo, Texture


class VelocityArrowField(Shader):
    """Visualize velocity field using procedural arrows.

    Samples velocity field on a grid and draws arrows in fragment shader.
    """

    def use(self, target_fbo: Fbo, velocity_tex: Texture, scale: float = 1.0,
            spacing: float = 8.0, arrow_length: float = 8.0, arrow_thickness: float = 0.8) -> None:
        """Render arrow field visualization to FBO.

        Args:
            target_fbo: Target framebuffer
            velocity_tex: Velocity texture (RG = XY velocity)
            scale: Velocity magnitude scale
            grid_spacing: Distance between arrow centers in pixels
            arrow_scale: Arrow length in pixels (e.g., 50 = 50 pixel long arrows)
            arrow_thickness: Arrow line thickness in pixels
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
        glUniform1f(glGetUniformLocation(self.shader_program, "grid_spacing"), spacing)
        glUniform1f(glGetUniformLocation(self.shader_program, "arrow_length"), arrow_length)
        glUniform1f(glGetUniformLocation(self.shader_program, "arrow_thickness"), arrow_thickness)

        # Render
        draw_quad()

        # Cleanup
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glUseProgram(0)
