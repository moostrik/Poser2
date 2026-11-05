"""VorticityForce3D - 3D vorticity confinement force."""
from __future__ import annotations

from OpenGL.GL import GL_RGBA16F, GL_TEXTURE2, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUniform3f, glUniform3i, glUseProgram

from modules.gl import Texture, Texture3D, ComputeShader


class VorticityForce3D(ComputeShader):
    """Compute 3D vorticity confinement force: F = ε * (N × ω) * dt."""

    WORKGROUP_SIZE_X = 16
    WORKGROUP_SIZE_Y = 16
    WORKGROUP_SIZE_Z = 1

    def __init__(self) -> None:
        super().__init__()

    def use(self, curl: Texture3D, obstacle: Texture3D, force_out: Texture3D,
            grid_scale: float, aspect: float, z_spacing: float,
            timestep: float, has_obstacles: bool = True,
            spin: Texture | None = None, spin_step: float = 0.0,
            spin_layer: float = 0.0, spin_spread: float = 0.001) -> None:
        """Compute vorticity confinement force.

        Args:
            curl: 3D curl vector field (RGBA16F)
            obstacle: 3D obstacle mask (R8)
            force_out: Output force volume (RGBA16F)
            grid_scale: Grid spacing
            aspect: Width/height ratio
            z_spacing: Sim-grid Z cell size in sim texels (dz/dx)
            timestep: Vorticity strength * dt
            has_obstacles: Whether obstacle logic is active
            spin: 2D curl (R16F) added as force at the injection depth; None = no spin input
            spin_step: Spin strength * dt
            spin_layer: Normalized depth [0, 1] of the injection
            spin_spread: Gaussian sigma in normalized depth units
        """
        if not self.allocated or not self.shader_program:
            return

        glUseProgram(self.shader_program)

        self.bind_texture_3d(0, curl, "uCurl")
        self.bind_texture_3d(1, obstacle, "uObstacle")
        self.bind_image_3d_write(0, force_out, GL_RGBA16F)

        glUniform1f(self.get_uniform_loc("uSpinStep"), spin_step if spin is not None else 0.0)
        if spin is not None:
            glActiveTexture(GL_TEXTURE2)
            glBindTexture(GL_TEXTURE_2D, spin.tex_id)
            glUniform1i(self.get_uniform_loc("uSpin"), 2)
            glUniform1f(self.get_uniform_loc("uSpinLayer"), spin_layer)
            glUniform1f(self.get_uniform_loc("uSpinSpread"), spin_spread)

        dx = grid_scale
        dy = grid_scale * aspect
        dz = grid_scale * z_spacing
        glUniform3f(self.get_uniform_loc("uHalfRdxInv"), 0.5 / dx, 0.5 / dy, 0.5 / dz)
        glUniform1f(self.get_uniform_loc("uTimestep"), timestep)
        glUniform3i(self.get_uniform_loc("uSize"),
                    force_out.width, force_out.height, force_out.depth)
        glUniform1i(self.get_uniform_loc("uHasObstacles"), int(has_obstacles))

        self.dispatch(force_out.width, force_out.height, force_out.depth)
