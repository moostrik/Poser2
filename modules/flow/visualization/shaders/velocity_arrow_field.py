"""Velocity Arrow Field shader.

Visualizes velocity field using procedural arrows without geometry shaders.
Ported from ofxFlowTools visualization concepts.
"""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, GL_VIEWPORT, glActiveTexture, glBindTexture, glGetIntegerv, glUniform1f, glUniform1i, glUniform2f, glUseProgram
from modules.gl import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)


class VelocityArrowField(Shader):
    """Visualize velocity field using procedural arrows.

    Samples velocity field on a grid and draws arrows in fragment shader: white arrows for a 2D
    velocity; for a 3D one, a line for x/y with a dot at its end growing with z, red toward the
    camera and green away.
    """

    def use(self, velocity_tex: Texture, scale: float = 1.0,
            spacing: float = 8.0, arrow_length: float = 8.0, arrow_thickness: float = 0.8,
            xyz: bool = False, dot_min: float = 0.0) -> None:
        """Render arrow field visualization to FBO.

        Args:
            velocity_tex: Velocity texture (RG = XY velocity, B = z with xyz)
            scale: Velocity magnitude scale
            spacing: Distance between arrow centers in pixels
            arrow_length: Arrow length in pixels at a velocity of 1
            arrow_thickness: Arrow line thickness in pixels
            xyz: Draw a 3D velocity: a line with a dot that grows with z
            dot_min: Smallest dot radius in pixels, with xyz
        """
        if not self.allocated or not self.shader_program:
            logger.warning("VelocityArrowField shader not allocated or shader program missing.")
            return
        if not velocity_tex.allocated:
            logger.warning("VelocityArrowField shader: input texture not allocated.")
            return

        viewport = glGetIntegerv(GL_VIEWPORT)
        output_width = viewport[2]
        output_height = viewport[3]

        # Activate shader program
        glUseProgram(self.shader_program)

        # Bind input texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, velocity_tex.tex_id)

        # Configure shader uniforms (using cached locations)
        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform1i(self.get_uniform_loc("xyz"), 1 if xyz else 0)
        glUniform1f(self.get_uniform_loc("scale"), scale)
        glUniform1f(self.get_uniform_loc("grid_spacing"), spacing)
        glUniform1f(self.get_uniform_loc("arrow_length"), arrow_length)
        glUniform1f(self.get_uniform_loc("arrow_thickness"), arrow_thickness)
        glUniform1f(self.get_uniform_loc("dot_min"), dot_min)
        glUniform2f(self.get_uniform_loc("resolution"), output_width, output_height)

        # Render
        draw_quad()
