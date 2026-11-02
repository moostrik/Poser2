"""Velocity Direction Map shader.

Visualizes velocity field using HSV color encoding where direction maps to hue; a 3D velocity as
x, y, z in red, green, blue around grey.
Ported from ofxFlowTools ftVelocityFieldShader.h
"""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)


class VelocityDirectionMap(Shader):
    """Visualize velocity field using HSV color encoding.

    Direction -> Hue, Magnitude -> Saturation/Value; with xyz, x/y/z -> red/green/blue around grey
    """

    def use(self, velocity_tex: Texture, scale: float = 1.0, xyz: bool = False) -> None:
        """Render velocity visualization to FBO.

        Args:
            velocity_tex: Velocity texture (RG = XY velocity, B = z with xyz)
            scale: Velocity scale multiplier
            xyz: Draw x, y, z as red, green, blue around grey
        """
        if not self.allocated or not self.shader_program:
            logger.warning("VelocityDirectionMap shader not allocated or shader program missing.")
            return
        if not velocity_tex.allocated:
            logger.warning("VelocityDirectionMap shader: input texture not allocated.")
            return

        # Activate shader program
        glUseProgram(self.shader_program)

        # Bind input texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, velocity_tex.tex_id)

        # Configure shader uniforms (using cached locations)
        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform1f(self.get_uniform_loc("scale"), scale)
        glUniform1i(self.get_uniform_loc("xyz"), 1 if xyz else 0)

        # Render
        draw_quad()
