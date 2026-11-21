"""MacCormack3D - the MacCormack correction of a semi-Lagrangian advection, clamped (Selle et al. 2008)."""
from __future__ import annotations

from OpenGL.GL import GL_RGBA16F, glUniform1f, glUniform1i, glUniform3f, glUniform3i, glUseProgram

from modules.gl import Texture3D, ComputeShader
from ...finite_guard import GuardSlot


class MacCormack3D(ComputeShader):
    """Correct a semi-Lagrangian predictor by half its error, traced forward again, clamped to the source's
    eight texels around the backward trace; then the dissipation and the evaporation."""

    WORKGROUP_SIZE_X = 8
    WORKGROUP_SIZE_Y = 8
    WORKGROUP_SIZE_Z = 4

    def __init__(self) -> None:
        super().__init__()

    def use(self, source: Texture3D, forward: Texture3D, result: Texture3D,
            velocity: Texture3D, obstacle: Texture3D,
            aspect: float, z_extent: float, timestep: float, dissipation: float,
            internal_format: int = GL_RGBA16F, has_obstacles: bool = True, evaporation: float = 0.0,
            evaporation_knee: float = 0.1, measure: int = 0,
            guard: GuardSlot | None = None,
            closed: tuple[bool, bool, bool] = (True, True, True), faces: Texture3D | None = None,
            clamp: bool = True) -> None:
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
            evaporation: dt / evaporation_time: a full value's share lost this step well above the knee, a
                saturating loss, every channel scaled with the size; 0 off
            evaporation_knee: The size below which the loss turns from a fixed amount to a share
            measure: How the size is read: 0 a premultiplied density's alpha, 1 a vector's length, 2 |r|
            guard: The field's slot in its stage's FiniteGuard: a non-finite value 0 and counted; None no guard
            closed: Per axis, whether the box's walls are closed (walls.glsl)
            faces: The velocity's faces (`FaceVelocity`), read instead of `velocity` for a field on the
                simulation's grid; None reads `velocity`
            clamp: Where the correction overshoots, cut it to the bounds instead of reverting to the predictor
        """
        if not self.allocated or not self.shader_program:
            return

        glUseProgram(self.shader_program)

        velocity.bind(0)
        source.bind(1)
        forward.bind(2)
        obstacle.bind(3)
        self.bind_image_3d_write(0, result, internal_format)

        glUniform1f(self.get_uniform_loc("uTimestep"), timestep)
        rdx_y = aspect          # a width per second of y crosses width / height of the uv
        rdx_z = 1.0 / z_extent if z_extent > 0.0 else 1.0
        glUniform3f(self.get_uniform_loc("uRdx"), 1.0, rdx_y, rdx_z)
        glUniform1f(self.get_uniform_loc("uDissipation"), dissipation)
        glUniform1f(self.get_uniform_loc("uEvaporation"), evaporation)
        glUniform1f(self.get_uniform_loc("uEvaporationKnee"), evaporation_knee)
        glUniform1i(self.get_uniform_loc("uMeasure"), measure)
        glUniform3i(self.get_uniform_loc("uClosed"), *(int(c) for c in closed))
        glUniform1i(self.get_uniform_loc("uFromFaces"), int(faces is not None))
        if faces is not None:
            faces.bind(4)
            glUniform3i(self.get_uniform_loc("uCells"), faces.width - 1, faces.height - 1, faces.depth - 1)
        if guard is not None:
            guard.use(self.get_uniform_loc("uGuardSlot"))
        else:
            glUniform1i(self.get_uniform_loc("uGuardSlot"), -1)
        glUniform3i(self.get_uniform_loc("uSize"), result.width, result.height, result.depth)
        glUniform1i(self.get_uniform_loc("uHasObstacles"), int(has_obstacles))
        glUniform1i(self.get_uniform_loc("uClamp"), int(clamp))

        self.dispatch(result.width, result.height, result.depth)
