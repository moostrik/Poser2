"""VorticityCurl3D - Compute 3D curl (vorticity vector) of velocity field."""
from __future__ import annotations

from OpenGL.GL import GL_RGBA16F, glUniform1f, glUniform1i, glUniform3f, glUniform3i, glUseProgram

from modules.gl import Texture3D, ComputeShader
from .face_binding import bind_faces


class VorticityCurl3D(ComputeShader):
    """Compute 3D curl: ω = ∇ × v (vorticity is a vector in 3D)."""

    WORKGROUP_SIZE_X = 8
    WORKGROUP_SIZE_Y = 8
    WORKGROUP_SIZE_Z = 4

    def __init__(self) -> None:
        super().__init__()

    def use(self, faces: Texture3D, obstacle: Texture3D,
            curl_out: Texture3D,
            cell: float,
            radius: float, has_obstacles: bool = True, closed: tuple[bool, bool, bool] = (True, True, True),
            wall_keep: float = 1.0) -> None:
        """Compute 3D velocity curl.

        Args:
            faces: The velocity's faces (`FaceVelocity`)
            obstacle: 3D obstacle mask (R8)
            curl_out: Output curl volume (RGBA16F: xyz = curl vector)
            cell: The cubic cell's size in widths
            radius: Curl sampling radius in texels
            has_obstacles: Whether obstacle logic is active
            closed: Per axis, whether the box's walls are closed (walls.glsl)
            wall_keep: The velocity's share along a closed wall kept a step; 1 free-slip
        """
        if not self.allocated or not self.shader_program:
            return

        glUseProgram(self.shader_program)

        cells = (curl_out.width, curl_out.height, curl_out.depth)
        bind_faces(self, faces, cells, closed, has_obstacles, obstacle)
        self.bind_image_3d_write(0, curl_out, GL_RGBA16F)

        dx = cell                         # cubic cells, in widths
        dy = dz = cell
        # Scale by radius: actual sample spacing is radius texels, not 1 texel
        # half_rdx = 0.5 / (dx * radius) to match the shader's sampling at uRadius texels
        glUniform3f(self.get_uniform_loc("uHalfRdxInv"),
                    0.5 / (dx * max(radius, 1e-6)),
                    0.5 / (dy * max(radius, 1e-6)),
                    0.5 / (dz * max(radius, 1e-6)))
        glUniform1f(self.get_uniform_loc("uRadius"), radius)
        glUniform1f(self.get_uniform_loc("uWallKeep"), wall_keep)

        self.dispatch(curl_out.width, curl_out.height, curl_out.depth)
