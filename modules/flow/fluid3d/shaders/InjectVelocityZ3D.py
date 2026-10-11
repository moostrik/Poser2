"""InjectVelocityZ3D - Inject signed Z-velocity from a 2D flow field into the velocity volume."""
from __future__ import annotations

from OpenGL.GL import *  # type: ignore

from modules.gl import Texture, Texture3D, ComputeShader


class InjectVelocityZ3D(ComputeShader):
    """Inject signed W (Z-velocity) from a 2D flow texture into a 3D velocity volume.

    W = clamp(squeeze * -div(flow) + push * |flow|, ±w_clamp), written additively
    into the gaussian depth slab. +W moves fluid away from the viewer.
    """

    WORKGROUP_SIZE_X = 16
    WORKGROUP_SIZE_Y = 16
    WORKGROUP_SIZE_Z = 1

    def __init__(self) -> None:
        super().__init__()

    def use(self, flow_2d: Texture, volume: Texture3D,
            target_layer: float, spread: float, strength: float,
            squeeze: float, push: float, w_clamp: float) -> None:
        """Inject signed Z-velocity into the velocity volume.

        Args:
            flow_2d: 2D flow velocity texture (RG16F, normalized-width units)
            volume: 3D velocity volume (read-write; W = .z)
            target_layer: Normalized depth [0, 1] — center of injection
            spread: Gaussian sigma in normalized depth units
            strength: Overall strength (input_strength_z * dt * motion)
            squeeze: Gain on -divergence (positive: looming pushes toward viewer)
            push: Gain on |flow| (positive = away from viewer)
            w_clamp: Abs clamp on the W source term
        """
        if not self.allocated or not self.shader_program:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, flow_2d.tex_id)
        glUniform1i(self.get_uniform_loc("uInput"), 0)

        self.bind_image_3d_readwrite(0, volume, GL_RGBA16F)

        glUniform1f(self.get_uniform_loc("uTargetLayer"), target_layer)
        glUniform1f(self.get_uniform_loc("uSpread"), spread)
        glUniform1f(self.get_uniform_loc("uStrength"), strength)
        glUniform1f(self.get_uniform_loc("uSqueeze"), squeeze)
        glUniform1f(self.get_uniform_loc("uPush"), push)
        glUniform1f(self.get_uniform_loc("uWClamp"), w_clamp)
        glUniform3i(self.get_uniform_loc("uSize"),
                    volume.width, volume.height, volume.depth)

        self.dispatch(volume.width, volume.height, volume.depth)
