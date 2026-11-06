"""MacCormack3D - the MacCormack correction of a semi-Lagrangian advection, clamped (Selle et al. 2008)."""
from __future__ import annotations

from OpenGL.GL import GL_RGBA16F, glUniform1f, glUniform1i, glUniform3f, glUniform3i, glUseProgram

from modules.gl import Texture3D, ComputeShader


class MacCormack3D(ComputeShader):
    """Correct a semi-Lagrangian predictor by half its error, traced forward again, clamped to the source's
    eight texels around the backward trace; then the dissipation and the evaporation."""

    WORKGROUP_SIZE_X = 16
    WORKGROUP_SIZE_Y = 16
    WORKGROUP_SIZE_Z = 1

    def __init__(self) -> None:
        super().__init__()

    def use(self, source: Texture3D, forward: Texture3D, result: Texture3D,
            velocity: Texture3D, obstacle: Texture3D,
            aspect: float, z_extent: float, timestep: float, dissipation: float,
            internal_format: int = GL_RGBA16F, has_obstacles: bool = True, evaporation: float = 0.0) -> None:
        """Write the corrected step into `result`.

        Args:
            source: The field before the step
            forward: The semi-Lagrangian predictor, the step without dissipation or evaporation
            result: Destination (the field's current buffer)
            velocity: 3D velocity field the step traced through
            obstacle: 3D obstacle mask
            aspect: Width over height: a width of y crosses this share of the uv
            z_extent: Slab thickness in sim-width units (uRdx.z = 1/z_extent)
            timestep: dt * speed
            dissipation: Decay multiplier
            internal_format: Image format for output
            has_obstacles: Whether obstacle logic is active
            evaporation: Amount subtracted this step, for the density; 0 for a signed field
        """
        if not self.allocated or not self.shader_program:
            return

        glUseProgram(self.shader_program)

        self.bind_texture_3d(0, velocity, "uVelocity")
        self.bind_texture_3d(1, source, "uSource")
        self.bind_texture_3d(2, forward, "uForward")
        self.bind_texture_3d(3, obstacle, "uObstacle")
        self.bind_image_3d_write(0, result, internal_format)

        glUniform1f(self.get_uniform_loc("uTimestep"), timestep)
        rdx_y = aspect          # a width per second of y crosses width / height of the uv
        rdx_z = 1.0 / z_extent if z_extent > 0.0 else 1.0
        glUniform3f(self.get_uniform_loc("uRdx"), 1.0, rdx_y, rdx_z)
        glUniform1f(self.get_uniform_loc("uDissipation"), dissipation)
        glUniform1f(self.get_uniform_loc("uEvaporation"), evaporation)
        glUniform3i(self.get_uniform_loc("uSize"), result.width, result.height, result.depth)
        glUniform1i(self.get_uniform_loc("uHasObstacles"), int(has_obstacles))

        self.dispatch(result.width, result.height, result.depth)
