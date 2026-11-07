"""VorticityForce3D - 3D vorticity confinement force."""
from __future__ import annotations

from OpenGL.GL import GL_RGBA16F, glUniform1f, glUniform1i, glUniform3f, glUniform3i, glUseProgram

from modules.gl import Texture3D, ComputeShader


class VorticityForce3D(ComputeShader):
    """The 3D vorticity confinement force, ε h (N × ω) dt, added to the velocity in place."""

    WORKGROUP_SIZE_X = 8
    WORKGROUP_SIZE_Y = 8
    WORKGROUP_SIZE_Z = 4

    def __init__(self) -> None:
        super().__init__()

    def use(self, curl: Texture3D, obstacle: Texture3D, velocity: Texture3D,
            cell: float,
            timestep: float, has_obstacles: bool = True) -> None:
        """Compute vorticity confinement force.

        Args:
            curl: 3D curl vector field (RGBA16F)
            obstacle: 3D obstacle mask (R8)
            velocity: The velocity the force is added to (RGBA16F, read-write)
            cell: The cubic cell's size in widths
            timestep: Vorticity strength * dt
            has_obstacles: Whether obstacle logic is active
        """
        if not self.allocated or not self.shader_program:
            return

        glUseProgram(self.shader_program)

        self.bind_texture_3d(0, curl, "uCurl")
        self.bind_texture_3d(1, obstacle, "uObstacle")
        self.bind_image_3d_readwrite(0, velocity, GL_RGBA16F)

        dx = cell                         # cubic cells, in widths
        dy = dz = cell
        glUniform3f(self.get_uniform_loc("uHalfRdxInv"), 0.5 / dx, 0.5 / dy, 0.5 / dz)
        glUniform1f(self.get_uniform_loc("uTimestep"), timestep)
        glUniform3i(self.get_uniform_loc("uSize"),
                    velocity.width, velocity.height, velocity.depth)
        glUniform1i(self.get_uniform_loc("uHasObstacles"), int(has_obstacles))

        self.dispatch(velocity.width, velocity.height, velocity.depth)
