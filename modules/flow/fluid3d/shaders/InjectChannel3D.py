"""InjectChannel3D - Inject single-channel 2D input into one RGBA channel of a 3D volume."""
from __future__ import annotations

from OpenGL.GL import *  # type: ignore

from modules.gl.Texture import Texture
from modules.gl.Texture3D import Texture3D
from modules.gl.ComputeShader import ComputeShader


class InjectChannel3D(ComputeShader):
    """Inject a single-channel 2D texture into one RGBA channel of a 3D volume.

    Uses gaussian depth spread, same as Inject3D, but only modifies the
    specified channel (R=0, G=1, B=2, A=3).
    """

    WORKGROUP_SIZE_X = 16
    WORKGROUP_SIZE_Y = 16
    WORKGROUP_SIZE_Z = 1

    def __init__(self) -> None:
        super().__init__()

    def use(self, input_2d: Texture, volume: Texture3D,
            target_layer: float, spread: float, channel: int,
            strength: float = 1.0, mode: int = 0,
            internal_format: int = GL_RGBA16F) -> None:
        """Inject single-channel 2D input into one channel of a 3D volume.

        Args:
            input_2d: 2D source texture (reads .r component)
            volume: 3D destination volume (read-write)
            target_layer: Normalized depth [0, 1] — center of injection
            spread: Gaussian sigma in normalized depth units
            channel: Target RGBA channel (0=R, 1=G, 2=B, 3=A)
            strength: Injection strength multiplier
            mode: 0 = additive, 1 = set (replace channel only)
            internal_format: Image format for volume binding
        """
        if not self.allocated or not self.shader_program:
            return

        glUseProgram(self.shader_program)

        # Bind 2D input as sampler
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, input_2d.tex_id)
        glUniform1i(self.get_uniform_loc("uInput"), 0)

        # Bind 3D volume as read-write image
        self.bind_image_3d_readwrite(0, volume, internal_format)

        # Uniforms
        glUniform1f(self.get_uniform_loc("uTargetLayer"), target_layer)
        glUniform1f(self.get_uniform_loc("uSpread"), spread)
        glUniform1f(self.get_uniform_loc("uStrength"), strength)
        glUniform1i(self.get_uniform_loc("uChannel"), channel)
        glUniform1i(self.get_uniform_loc("uMode"), mode)
        glUniform3i(self.get_uniform_loc("uSize"),
                    volume.width, volume.height, volume.depth)

        self.dispatch(volume.width, volume.height, volume.depth)
