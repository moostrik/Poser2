"""BoundaryVelocity3D - the boundary: the velocity volume dragged toward the body's velocity in the birth slab."""
from __future__ import annotations

from OpenGL.GL import GL_RGBA16F, GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUniform3i, glUseProgram

from modules.gl import Texture, Texture3D, ComputeShader


class BoundaryVelocity3D(ComputeShader):
    """Drag the velocity volume toward the body's velocity where the body is, never past it:
    v ← v + (v_body × gain − v) × share × gate × weight × slab. x and y are the image plane's; the
    volume's w is the velocity's −z, since z points toward the camera and +W away from the viewer."""

    WORKGROUP_SIZE_X = 8
    WORKGROUP_SIZE_Y = 8
    WORKGROUP_SIZE_Z = 4

    def __init__(self) -> None:
        super().__init__()

    def use(self, velocity: Texture, volume: Texture3D, target_layer: float, spread: float,
            gain: float, share: float, weight: float, floor: float) -> None:
        """Drag the volume toward the body's velocity.

        Args:
            velocity: The body's velocity (RGBA16F: x, y, z toward the camera; widths/s)
            volume: 3D velocity volume (read-write; u,v,w = .xyz)
            target_layer: Normalized depth [0, 1] — center of the slab
            spread: Gaussian sigma in normalized depth units
            gain: The body's velocity times this is the target
            share: 1 − exp(−dt / drag_time), the step's share of the gap
            weight: The injection's weight, 0 to 1
            floor: widths/s, the body's speed below which it drags nothing
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
        glUniform1f(self.get_uniform_loc("uGain"), gain)
        glUniform1f(self.get_uniform_loc("uShare"), share)
        glUniform1f(self.get_uniform_loc("uWeight"), weight)
        glUniform1f(self.get_uniform_loc("uFloor"), floor)
        glUniform3i(self.get_uniform_loc("uSize"),
                    volume.width, volume.height, volume.depth)

        self.dispatch(volume.width, volume.height, volume.depth)
