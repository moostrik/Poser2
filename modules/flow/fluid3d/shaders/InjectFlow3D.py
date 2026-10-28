"""InjectFlow3D - Inject the 2D flow as a full 3D velocity (u, v measured; w synthesized)."""
from __future__ import annotations

from OpenGL.GL import *  # type: ignore

from modules.gl import Texture, Texture3D, ComputeShader


class InjectFlow3D(ComputeShader):
    """Inject the 2D flow texture into the 3D velocity volume as one vec3 input.

    (u, v) are the measured flow; w is synthesized from the same flow as
    push * |flow| + squeeze * -divergence (clamped). All three components are
    written into the gaussian birth slab and treated identically from then on.
    +W moves fluid away from the viewer.
    """

    WORKGROUP_SIZE_X = 16
    WORKGROUP_SIZE_Y = 16
    WORKGROUP_SIZE_Z = 1

    def __init__(self) -> None:
        super().__init__()

    def use(self, flow_2d: Texture, volume: Texture3D,
            target_layer: float, spread: float,
            strength_xy: float, strength_w: float,
            push: float, squeeze: float, w_clamp: float) -> None:
        """Inject the flow into the velocity volume.

        Args:
            flow_2d: 2D flow velocity texture (RG16F, normalized-width units)
            volume: 3D velocity volume (read-write; u,v,w = .xyz)
            target_layer: Normalized depth [0, 1] — center of injection
            spread: Gaussian sigma in normalized depth units
            strength_xy: Gain on the measured components (strength * input_strength * dt)
            strength_w: Gain on the synthesized component (strength * dt)
            push: W synthesis — gain on |flow| (signed; + = away from viewer)
            squeeze: W synthesis — gain on -divergence (signed)
            w_clamp: Abs clamp on the synthesized W
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
        glUniform1f(self.get_uniform_loc("uStrengthXY"), strength_xy)
        glUniform1f(self.get_uniform_loc("uStrengthW"), strength_w)
        glUniform1f(self.get_uniform_loc("uPush"), push)
        glUniform1f(self.get_uniform_loc("uSqueeze"), squeeze)
        glUniform1f(self.get_uniform_loc("uWClamp"), w_clamp)
        glUniform3i(self.get_uniform_loc("uSize"),
                    volume.width, volume.height, volume.depth)

        self.dispatch(volume.width, volume.height, volume.depth)
