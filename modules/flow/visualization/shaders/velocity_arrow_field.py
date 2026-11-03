"""Velocity Arrow Field shader.

Visualizes velocity field using procedural arrows without geometry shaders.
Ported from ofxFlowTools visualization concepts.
"""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, GL_VIEWPORT, glActiveTexture, glBindTexture, glGetIntegerv, glUniform1f, glUniform1i, glUniform2f, glUseProgram
from modules.gl import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)


class VelocityArrowField(Shader):
    """Visualize velocity field using procedural markers.

    Samples velocity field on a grid and draws one outlined blob per grid point in the fragment
    shader: a round body for z, red toward the camera and green away, and a white bulb on a stem
    reaching out along x/y. Its size grows with the speed, on the cell, rolling off toward 2 cells.
    """

    def use(self, velocity_tex: Texture, scale: float = 1.0,
            spacing: float = 8.0, element_length: float = 1.0, arrow_thickness: float = 0.8,
            xyz: bool = False) -> None:
        """Render arrow field visualization to FBO.

        Args:
            velocity_tex: Velocity texture (RG = XY velocity, B = z with xyz)
            scale: Velocity magnitude scale
            spacing: Distance between grid points in pixels
            element_length: The cell a blob is sized on, a ratio of spacing
            arrow_thickness: Outline thickness in pixels
            xyz: Draw a 3D velocity: z as the blob's body
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
        glUniform1f(self.get_uniform_loc("element_length"), element_length)
        glUniform1f(self.get_uniform_loc("arrow_thickness"), arrow_thickness)
        glUniform2f(self.get_uniform_loc("resolution"), output_width, output_height)

        # Render
        draw_quad()
