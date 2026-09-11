"""Unwrap a ring of camera frames into one 360-degree strip."""

from OpenGL.GL import *  # type: ignore
from modules.gl import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)

MAX_CAMS: int = 8  # must match the #define in panoramicstitch.frag


class PanoramicStitch(Shader):
    """Draw every camera into one azimuth strip, in a single pass.

    One pass rather than one draw per camera: the fragment shader knows how many cameras cover
    each column, so an averaging blend divides by the true coverage instead of a blend-equation
    approximation, and no blend state has to be set up and torn down.
    """

    def use(self, textures: list[Texture], cam_fov: float, target_fov: float,
            ring_radius: float, focus_diameter: float, average: bool) -> None:
        """Args:
            textures: one per camera, in camera-id order; camera 0 owns azimuth 0 upward
            cam_fov: one camera's horizontal field (degrees)
            target_fov: the sector one camera owns, 360 / num_cameras (degrees)
            ring_radius: camera distance from the rig centre (m); 0 disables the parallax term
            focus_diameter: the play-zone cylinder the image is aligned for (m)
            average: True averages the overlap, False takes the brighter of the two
        """
        if not self.allocated or not self.shader_program:
            logger.warning("PanoramicStitch shader not allocated or shader program missing.")
            return
        if not textures:
            return

        glUseProgram(self.shader_program)

        # Bind what we have; point the spare sampler slots at camera 0 so no unit is ever
        # unbound, even though the shader's loop stops at numCams.
        num_cams: int = min(len(textures), MAX_CAMS)
        for unit in range(MAX_CAMS):
            source: Texture = textures[unit] if unit < num_cams else textures[0]
            glActiveTexture(GL_TEXTURE0 + unit)
            glBindTexture(GL_TEXTURE_2D, source.tex_id if source.allocated else 0)
            glUniform1i(self.get_uniform_loc(f"tex[{unit}]"), unit)

        glUniform1i(self.get_uniform_loc("numCams"), num_cams)
        glUniform1f(self.get_uniform_loc("camFov"), cam_fov)
        glUniform1f(self.get_uniform_loc("targetFov"), target_fov)
        glUniform1f(self.get_uniform_loc("ringRadius"), ring_radius)
        glUniform1f(self.get_uniform_loc("focusRadius"), focus_diameter / 2.0)
        glUniform1i(self.get_uniform_loc("blendMode"), 1 if average else 0)

        draw_quad()

        glActiveTexture(GL_TEXTURE0)
