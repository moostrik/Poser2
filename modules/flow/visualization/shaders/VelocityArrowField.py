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
            grid_spacing: float = 20.0, arrow_scale: float = 1.0,
            arrow_thickness: float = 0.15) -> None:
        """Render arrow field visualization to FBO.

        Args:
            target_fbo: Target framebuffer
            velocity_tex: Velocity texture (RG = XY velocity)
            scale: Velocity magnitude scale
            grid_spacing: Distance between arrow centers in pixels
            arrow_scale: Arrow size relative to grid (0.8 = 80% of grid cell)
            arrow_thickness: Arrow line thickness (0-1)
        """
        if not self.allocated:
            return
        if self.shader_program is None:
            return

        glBindFramebuffer(GL_FRAMEBUFFER, target_fbo.fbo_id)
        glViewport(0, 0, target_fbo.width, target_fbo.height)

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, velocity_tex.tex_id)
        glUniform1i(glGetUniformLocation(self.shader_program, "tex0"), 0)
        glUniform1f(glGetUniformLocation(self.shader_program, "scale"), scale)
        glUniform1f(glGetUniformLocation(self.shader_program, "grid_spacing"), grid_spacing)
        glUniform1f(glGetUniformLocation(self.shader_program, "arrow_scale"), arrow_scale)
        glUniform1f(glGetUniformLocation(self.shader_program, "arrow_thickness"), arrow_thickness)
        glUniform2f(glGetUniformLocation(self.shader_program, "resolution"),
                    float(target_fbo.width), float(target_fbo.height))

        draw_quad()

        glUseProgram(0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
