"""InjectFlow3D - Inject the 3D velocity (x, y on the image plane, z toward the camera) into the velocity volume."""
from __future__ import annotations

from OpenGL.GL import GL_RGBA16F, GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUniform3f, glUniform3i, glUseProgram

from modules.gl import Texture, Texture3D, ComputeShader


class InjectFlow3D(ComputeShader):
    """Inject the 3D velocity texture into the 3D velocity volume as one vec3 input.

    x and y are the image plane's; the volume's w is the velocity's -z, since z points toward the
    camera and +W away from the viewer. All three are written into the gaussian birth slab and
    treated identically from then on.
    """

    WORKGROUP_SIZE_X = 16
    WORKGROUP_SIZE_Y = 16
    WORKGROUP_SIZE_Z = 1

    def __init__(self) -> None:
        super().__init__()

    def use(self, velocity: Texture, volume: Texture3D, target_layer: float, spread: float,
            strength: tuple[float, float, float]) -> None:
        """Inject the velocity into the velocity volume.

        Args:
            velocity: 3D velocity texture (RGBA16F: x, y, z toward the camera; normalized-width units)
            volume: 3D velocity volume (read-write; u,v,w = .xyz)
            target_layer: Normalized depth [0, 1] — center of injection
            spread: Gaussian sigma in normalized depth units
            strength: Gain on x, y and z (strength * gain * input_x, input_y, 1 * dt)
        """
        if not self.allocated or not self.shader_program:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, velocity.tex_id)
        glUniform1i(self.get_uniform_loc("uInput"), 0)

        self.bind_image_3d_readwrite(0, volume, GL_RGBA16F)

        glUniform1f(self.get_uniform_loc("uTargetLayer"), target_layer)
        glUniform1f(self.get_uniform_loc("uSpread"), spread)
        glUniform3f(self.get_uniform_loc("uStrength"), *strength)
        glUniform3i(self.get_uniform_loc("uSize"),
                    volume.width, volume.height, volume.depth)

        self.dispatch(volume.width, volume.height, volume.depth)
