"""VorticityForce3D - 3D vorticity confinement force."""
from __future__ import annotations

from OpenGL.GL import *  # type: ignore

from modules.gl import Texture3D, ComputeShader


class VorticityForce3D(ComputeShader):
    """Compute 3D vorticity confinement force: F = ε * (N × ω) * dt."""

    WORKGROUP_SIZE_X = 16
    WORKGROUP_SIZE_Y = 16
    WORKGROUP_SIZE_Z = 1

    def __init__(self) -> None:
        super().__init__()

    def use(self, curl: Texture3D, obstacle: Texture3D, force_out: Texture3D,
            grid_scale: float, aspect: float, depth_scale: float,
            timestep: float, has_obstacles: bool = True) -> None:
        """Compute vorticity confinement force.

        Args:
            curl: 3D curl vector field (RGBA16F)
            obstacle: 3D obstacle mask (R8)
            force_out: Output force volume (RGBA16F)
            grid_scale: Grid spacing
            aspect: Width/height ratio
            depth_scale: Z grid spacing relative to XY
            timestep: Vorticity strength * dt
            has_obstacles: Whether obstacle logic is active
        """
        if not self.allocated or not self.shader_program:
            return

        glUseProgram(self.shader_program)

        self.bind_texture_3d(0, curl, "uCurl")
        self.bind_texture_3d(1, obstacle, "uObstacle")
        self.bind_image_3d_write(0, force_out, GL_RGBA16F)

        dx = grid_scale
        dy = grid_scale * aspect
        dz = grid_scale * depth_scale
        glUniform3f(self.get_uniform_loc("uHalfRdxInv"), 0.5 / dx, 0.5 / dy, 0.5 / dz)
        glUniform1f(self.get_uniform_loc("uTimestep"), timestep)
        glUniform3i(self.get_uniform_loc("uSize"),
                    force_out.width, force_out.height, force_out.depth)
        glUniform1i(self.get_uniform_loc("uHasObstacles"), int(has_obstacles))

        self.dispatch(force_out.width, force_out.height, force_out.depth)
