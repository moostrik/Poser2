"""BodyForceAcceleration3D - the body force: an acceleration added × dt into the velocity volume's birth slab."""
from __future__ import annotations

from OpenGL.GL import GL_RGBA16F, GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUniform3i, glUseProgram

from modules.gl import Texture, Texture3D, ComputeShader


class BodyForceAcceleration3D(ComputeShader):
    """Add an acceleration × dt into the velocity volume's gaussian birth slab. x and y are the image
    plane's; the volume's w is the acceleration's −z, since z points toward the camera and +W away
    from the viewer."""

    WORKGROUP_SIZE_X = 8
    WORKGROUP_SIZE_Y = 8
    WORKGROUP_SIZE_Z = 4

    def __init__(self) -> None:
        super().__init__()

    def use(self, acceleration: Texture, volume: Texture3D, target_layer: float, spread: float,
            strength: float, depth_scale: float = 1.0) -> None:
        """Add the acceleration into the volume.

        Args:
            acceleration: The acceleration (RGBA16F: x, y, z toward the camera; widths/s²)
            volume: 3D velocity volume (read-write; u,v,w = .xyz)
            target_layer: Normalized depth [0, 1] — center of the slab
            spread: Gaussian sigma in normalized depth units
            strength: weight × gain × dt
            depth_scale: The acceleration's z times this is the water's, as the boundary's
        """
        if not self.allocated or not self.shader_program:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, acceleration.tex_id)
        glUniform1i(self.get_uniform_loc("uInput"), 0)

        self.bind_image_3d_readwrite(0, volume, GL_RGBA16F)

        glUniform1f(self.get_uniform_loc("uTargetLayer"), target_layer)
        glUniform1f(self.get_uniform_loc("uSpread"), spread)
        glUniform1f(self.get_uniform_loc("uStrength"), strength)
        glUniform1f(self.get_uniform_loc("uDepthScale"), depth_scale)
        glUniform3i(self.get_uniform_loc("uSize"),
                    volume.width, volume.height, volume.depth)

        self.dispatch(volume.width, volume.height, volume.depth)
