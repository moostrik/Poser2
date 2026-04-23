"""VorticityCurl3D - Compute 3D curl (vorticity vector) of velocity field."""
from __future__ import annotations

from OpenGL.GL import *  # type: ignore

from modules.gl import Texture3D, ComputeShader


class VorticityCurl3D(ComputeShader):
    """Compute 3D curl: ω = ∇ × v (vorticity is a vector in 3D)."""

    WORKGROUP_SIZE_X = 16
    WORKGROUP_SIZE_Y = 16
    WORKGROUP_SIZE_Z = 1

    def __init__(self) -> None:
        super().__init__()

    def use(self, velocity: Texture3D, obstacle: Texture3D,
            curl_out: Texture3D,
            grid_scale: float, aspect: float, depth_scale: float,
            radius: float, has_obstacles: bool = True) -> None:
        """Compute 3D velocity curl.

        Args:
            velocity: 3D velocity field (RGBA16F)
            obstacle: 3D obstacle mask (R8)
            curl_out: Output curl volume (RGBA16F: xyz = curl vector)
            grid_scale: Grid spacing
            aspect: Width/height ratio
            depth_scale: Z grid spacing relative to XY
            radius: Curl sampling radius in texels
            has_obstacles: Whether obstacle logic is active
        """
        if not self.allocated or not self.shader_program:
            return

        glUseProgram(self.shader_program)

        self.bind_texture_3d(0, velocity, "uVelocity")
        self.bind_texture_3d(1, obstacle, "uObstacle")
        self.bind_image_3d_write(0, curl_out, GL_RGBA16F)

        dx = grid_scale
        dy = grid_scale * aspect
        dz = grid_scale * depth_scale
        # Scale by radius: actual sample spacing is radius texels, not 1 texel
        # half_rdx = 0.5 / (dx * radius) to match the shader's sampling at uRadius texels
        glUniform3f(self.get_uniform_loc("uHalfRdxInv"),
                    0.5 / (dx * max(radius, 1e-6)),
                    0.5 / (dy * max(radius, 1e-6)),
                    0.5 / (dz * max(radius, 1e-6)))
        glUniform1f(self.get_uniform_loc("uRadius"), radius)
        glUniform3i(self.get_uniform_loc("uSize"),
                    curl_out.width, curl_out.height, curl_out.depth)
        glUniform1i(self.get_uniform_loc("uHasObstacles"), int(has_obstacles))

        self.dispatch(curl_out.width, curl_out.height, curl_out.depth)
