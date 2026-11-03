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
    velocity; for a 3D one, the white arrow for x/y with a ring around the grid point for z, red
    toward the camera and green away. Sizes are on the cell: at a speed of 1 the arrow reaches 1 cell
    and the ring's radius is 1/pi cell, the same length of line; faster, both roll off toward 2 cells.
    """

    def use(self, velocity_tex: Texture, scale: float = 1.0, z_scale: float = 1.0,
            spacing: float = 8.0, element_length: float = 1.0, arrow_thickness: float = 0.8,
            xyz: bool = False) -> None:
        """Render arrow field visualization to FBO.

        Args:
            velocity_tex: Velocity texture (RG = XY velocity, B = z with xyz)
            scale: Velocity magnitude scale
            z_scale: Multiplier on z alone
            spacing: Distance between grid points in pixels
            element_length: The cell the arrows are sized on, a ratio of spacing
            arrow_thickness: Arrow line thickness in pixels
            xyz: Draw a 3D velocity: the arrow with a ring for z
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
        glUniform1f(self.get_uniform_loc("z_scale"), z_scale)
        glUniform1f(self.get_uniform_loc("grid_spacing"), spacing)
        glUniform1f(self.get_uniform_loc("element_length"), element_length)
        glUniform1f(self.get_uniform_loc("arrow_thickness"), arrow_thickness)
        glUniform2f(self.get_uniform_loc("resolution"), output_width, output_height)

        # Render
        draw_quad()
