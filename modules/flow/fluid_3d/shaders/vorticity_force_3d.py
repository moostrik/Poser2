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

    def use(self, curl: Texture3D, obstacle: Texture3D, force_out: Texture3D,
            cell: tuple[float, float, float],
            timestep: float, has_obstacles: bool = True, curl_2: Texture3D | None = None, timestep_2: float = 0.0) -> None:
        """Compute vorticity confinement force.

        Args:
            curl: 3D curl vector field (RGBA16F)
            obstacle: 3D obstacle mask (R8)
            force_out: The force × dt at the cells (RGBA16F), for AddToFaces3D
            cell: A cell's sides in widths, x, y and z
            timestep: Vorticity strength * dt
            has_obstacles: Whether obstacle logic is active
            curl_2: A second confinement's curl, its force added with `timestep_2`; None for one
        """
        if not self.allocated or not self.shader_program:
            return

        glUseProgram(self.shader_program)

        self.bind_texture_3d(0, curl, "uCurl")
        self.bind_texture_3d(1, obstacle, "uObstacle")
        self.bind_image_3d_write(0, force_out, GL_RGBA16F)

        glUniform3f(self.get_uniform_loc("uHalfRdxInv"), 0.5 / cell[0], 0.5 / cell[1], 0.5 / cell[2])
        glUniform1f(self.get_uniform_loc("uTimestep"), timestep)
        if curl_2 is not None:
            self.bind_texture_3d(2, curl_2, "uCurl2")
        glUniform1i(self.get_uniform_loc("uSecond"), int(curl_2 is not None))
        glUniform1f(self.get_uniform_loc("uTimestep2"), timestep_2)
        glUniform3i(self.get_uniform_loc("uSize"),
                    force_out.width, force_out.height, force_out.depth)
        glUniform1i(self.get_uniform_loc("uHasObstacles"), int(has_obstacles))

        self.dispatch(force_out.width, force_out.height, force_out.depth)
