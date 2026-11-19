"""Fluid2D - the 2D Navier-Stokes fluid, its passes and its fields on the plane, as the 3D's for what a plane has.
How and why: modules/flow/FLUID_2D.md."""
from OpenGL.GL import GL_CLAMP_TO_BORDER, GL_CLAMP_TO_EDGE, GL_R16F, GL_R32F, GL_R8, GL_RG16F, GL_RGBA16F, \
    GL_SHADER_IMAGE_ACCESS_BARRIER_BIT, GL_TEXTURE_FETCH_BARRIER_BIT, glMemoryBarrier

from modules.gl import Texture, SwapFbo, Fbo, GpuTimer, StorageBuffer
from .. import FlowUtil, Port, TextureKind
from ..density_blend import DensityBlend
from ..drag import DRAG_FLOOR, drag_share
from ..finite_guard import FiniteGuard, GuardSlot
from ..advection import Advection, Limiter
from ..port import ColorSpace
from ..pressure_solver import PressureSolver
from .. import pressure_reach
from ..time_step import REFERENCE_STEP, TimeStep, viscosity_iterations
from ..shaders import AddBoolean
from ..wall_mode import WallMode
from .fluid_settings import Fluid2DSettings, InputSettings, PressureSettings
from .multigrid_pressure_2d import MultigridPressure2D
from .shaders import (
    Advect, Divergence, Gradient,
    JacobiPressure, JacobiPressureCompute, JacobiDiffusion, JacobiDiffusionCompute,
    VorticityCurl, VorticityForce, Buoyancy,
    InjectVelocity2D, InjectDensity2D, InjectScalar2D, ScalarInjection, DensityToRgb2D, MacCormack2D,
    ExpansionMean2D, PARTIAL_BYTES
)

_BARRIER_FETCH_AND_IMAGE: int = int(GL_TEXTURE_FETCH_BARRIER_BIT) | int(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

_WALL_GL: dict = {
    WallMode.OPEN:   (GL_CLAMP_TO_BORDER, (0.0, 0.0, 0.0, 0.0)),
    WallMode.CLOSED: (GL_CLAMP_TO_EDGE,   (0.0, 0.0, 0.0, 0.0)),
}


# ---------------------------------------------------------------------------
# 2D Fluid Simulation
# ---------------------------------------------------------------------------

class Fluid2D:
    """2D Navier-Stokes fluid simulation.

    Does NOT inherit FlowBase -- manages its own SwapFbo ping-pong
    buffers directly.  Provides a compatible public API with the 3D
    Fluid3D where applicable.

    Fields:
        velocity:             SwapFbo RG16F    (u, v)
        density:              SwapFbo RGBA16F   (r, g, b, a)
        temperature:          SwapFbo R16F
        pressure:             SwapFbo R32F
        simulation_obstacle:  SwapFbo R8        (sim resolution)
        density_obstacle:     SwapFbo R8        (density resolution)

    Intermediate (single-buffer):
        divergence:       Fbo R16F
        vorticity_curl:   Fbo RGBA16F  (a curl per confinement)
        vorticity_force:  Fbo RG16F
        buoyancy:         Fbo RG16F
    """

    def __init__(self, settings: Fluid2DSettings | None = None) -> None:
        self.settings: Fluid2DSettings = settings or Fluid2DSettings()

        # ---- Simulation dimensions and state ---
        self._simulation_width: int = 0
        self._simulation_height: int = 0
        self._density_width: int = 0
        self._density_height: int = 0
        self._field_aspect: float = 1.0    # the field's width over its height, the density's: one width is this of the height
        self._cell: tuple[float, float] = (1.0, 1.0)   # a simulation cell's width and height, widths

        self._dt: float = REFERENCE_STEP
        self._time_step: TimeStep = TimeStep()

        self._has_obstacles: bool = False

        self._allocated: bool = False

        self._reallocate_pending: bool = False
        self._wall_dirty: bool = False
        self._temperature_clear: bool = True    # the temperature cleared since it last ran

        # ---- Simulation fields (SwapFbo for ping-pong) ----
        # The velocity, the density and the temperature read beyond the walls by `wall` (_apply_wall_modes()); every pass
        # that reads a neighbor there reads the walls' ghosts (walls_2d.glsl) instead
        self._velocity_fbo: SwapFbo = SwapFbo(wrap=GL_CLAMP_TO_BORDER, border_color=(0.0, 0.0, 0.0, 0.0))
        self._density_fbo: SwapFbo = SwapFbo()
        self._temperature_fbo: SwapFbo = SwapFbo()
        self._pressure_fbo: SwapFbo = SwapFbo()
        # Obstacle: CLAMP_TO_BORDER(1) = out-of-bounds = obstacle
        self._simulation_obstacle_fbo: SwapFbo = SwapFbo(wrap=GL_CLAMP_TO_BORDER, border_color=(1.0, 1.0, 1.0, 1.0))
        # Full-resolution obstacle for density advection (same wrap)
        self._density_obstacle_fbo: SwapFbo = SwapFbo(wrap=GL_CLAMP_TO_BORDER, border_color=(1.0, 1.0, 1.0, 1.0))

        # Intermediate result FBOs (single buffer, no ping-pong)
        self._divergence_fbo: Fbo = Fbo()
        self._expansion_fbo: SwapFbo = SwapFbo()     # the expansion over this tick's injections
        # with closed walls its mean over the water's cells, per workgroup, then all (ExpansionMean2D)
        self._expansion_partials: StorageBuffer = StorageBuffer()
        self._expansion_mean: StorageBuffer = StorageBuffer()
        self._expansion_fresh: bool = False          # this tick's first injection has cleared it
        self._picture_fbo: Fbo = Fbo()               # the density as RGB, premultiplied, where it is held as Oklab
        self._oklab: bool = self.settings.density.color_space == ColorSpace.OKLAB
        # Curl is sampled at neighbor offsets by VorticityForce: beyond the walls its edge, the cell's own, as the 3D's
        self._vorticity_curl_fbo: Fbo = Fbo(wrap=GL_CLAMP_TO_EDGE)
        self._vorticity_force_fbo: Fbo = Fbo()
        self._buoyancy_fbo: Fbo = Fbo()         # the buoyancy, for its monitor
        self._buoyancy_shown: bool = False      # a monitor read the buoyancy since the last step
        # MacCormack's predictors, one per field, each allocated at its field's first MacCormack step
        self._velocity_scratch: Fbo = Fbo(wrap=GL_CLAMP_TO_BORDER, border_color=(0.0, 0.0, 0.0, 0.0))
        self._temperature_scratch: Fbo = Fbo()
        self._density_scratch: Fbo = Fbo()

        # Shaders
        self._advect_shader: Advect = Advect()
        self._maccormack_shader: MacCormack2D = MacCormack2D()
        self._divergence_shader: Divergence = Divergence()
        self._expansion_mean_shader: ExpansionMean2D = ExpansionMean2D()
        self._gradient_shader: Gradient = Gradient()
        self._jacobi_pressure_shader: JacobiPressure = JacobiPressure()
        self._jacobi_pressure_compute: JacobiPressureCompute = JacobiPressureCompute()
        self._jacobi_diffusion_shader: JacobiDiffusion = JacobiDiffusion()
        self._jacobi_diffusion_compute: JacobiDiffusionCompute = JacobiDiffusionCompute()
        self._multigrid: MultigridPressure2D = MultigridPressure2D()     # SOR and the multigrid's levels

        # Use compute shaders for iterative solvers (faster)
        self._use_compute_pressure: bool = True
        self._use_compute_diffusion: bool = True
        self._vorticity_curl_shader: VorticityCurl = VorticityCurl()
        self._vorticity_force_shader: VorticityForce = VorticityForce()
        self._buoyancy_shader: Buoyancy = Buoyancy()
        self._add_boolean_shader: AddBoolean = AddBoolean()
        self._inject_velocity_shader: InjectVelocity2D = InjectVelocity2D()
        self._inject_density_shader: InjectDensity2D = InjectDensity2D()
        self._inject_scalar_shader: InjectScalar2D = InjectScalar2D()
        self._picture_shader: DensityToRgb2D = DensityToRgb2D()
        self._gpu_timer: GpuTimer = GpuTimer("GPU fluid")
        self._guard: FiniteGuard = FiniteGuard('fluid', ('velocity', 'density', 'temperature'))

        # Bind settings actions
        self.settings.bind(Fluid2DSettings.width, lambda _: self._request_reallocate())
        self.settings.bind(Fluid2DSettings.height, lambda _: self._request_reallocate())
        self.settings.bind(Fluid2DSettings.wall, lambda _: setattr(self, '_wall_dirty', True))

    # ========== Allocation ==========

    def allocate(self, width: int, height: int) -> None:
        """Allocate all FBOs and shaders.

        Args:
            width: Full output resolution width
            height: Full output resolution height
        """
        self._density_width = width
        self._density_height = height
        self._update_simulation_dimensions()

        self._allocate_simulation_fields()
        self._allocate_density_fields()

        # Allocate shaders
        self._advect_shader.allocate()
        self._maccormack_shader.allocate()
        self._divergence_shader.allocate()
        self._expansion_mean_shader.allocate()
        self._gradient_shader.allocate()
        self._jacobi_pressure_shader.allocate()
        self._jacobi_pressure_compute.allocate()
        self._jacobi_diffusion_shader.allocate()
        self._jacobi_diffusion_compute.allocate()
        self._vorticity_curl_shader.allocate()
        self._vorticity_force_shader.allocate()
        self._buoyancy_shader.allocate()
        self._add_boolean_shader.allocate()
        for shader in self._input_shaders():
            shader.allocate()
        self._guard.allocate()

        self._allocated = True

    def _update_simulation_dimensions(self) -> None:
        """Recompute simulation dimensions from current settings."""
        self._simulation_width = self._align8(self.settings.width)
        self._simulation_height = self._align8(self.settings.height)
        self._field_aspect = self._density_width / self._density_height if self._density_height > 0 else 1.0
        # the field is one width across and 1 / field_aspect widths high, the grid's cells over it
        self._cell = (1.0 / max(self._simulation_width, 1), 1.0 / (self._field_aspect * max(self._simulation_height, 1)))

    def _allocate_density_fields(self) -> None:
        """(Re)allocate density-resolution FBOs (full output resolution)."""
        self._density_fbo.allocate(self._density_width, self._density_height, GL_RGBA16F)
        self._picture_fbo.allocate(self._density_width, self._density_height, GL_RGBA16F)
        self._density_obstacle_fbo.allocate(self._density_width, self._density_height, GL_R8)
        self._apply_wall_modes()

    def _allocate_simulation_fields(self) -> None:
        """(Re)allocate simulation-resolution FBOs."""
        sim_w = self._simulation_width
        sim_h = self._simulation_height

        # Primary simulation fields
        self._velocity_fbo.allocate(sim_w, sim_h, GL_RG16F)
        self._temperature_fbo.allocate(sim_w, sim_h, GL_R16F)
        self._temperature_clear = True
        self._pressure_fbo.allocate(sim_w, sim_h, GL_R32F)
        self._multigrid.allocate(sim_w, sim_h, self._cell)
        self._simulation_obstacle_fbo.allocate(sim_w, sim_h, GL_R8)
        FlowUtil.blit(self._simulation_obstacle_fbo, self._density_obstacle_fbo)

        # Intermediate FBOs
        self._divergence_fbo.allocate(sim_w, sim_h, GL_R16F)
        self._vorticity_curl_fbo.allocate(sim_w, sim_h, GL_RGBA16F)     # each confinement's curl, one channel each
        self._vorticity_force_fbo.allocate(sim_w, sim_h, GL_RG16F)
        self._buoyancy_fbo.allocate(sim_w, sim_h, GL_RG16F)
        self._expansion_fbo.allocate(sim_w, sim_h, GL_R16F)
        self._expansion_fbo.clear_all()
        self._expansion_partials.allocate(ExpansionMean2D.groups((sim_w, sim_h)) * PARTIAL_BYTES)
        self._expansion_mean.allocate(4)

        self._apply_wall_modes()

    def deallocate(self) -> None:
        """Release all GPU resources."""
        self._velocity_fbo.deallocate()
        self._density_fbo.deallocate()
        self._temperature_fbo.deallocate()
        self._pressure_fbo.deallocate()
        self._simulation_obstacle_fbo.deallocate()
        self._density_obstacle_fbo.deallocate()
        self._divergence_fbo.deallocate()
        self._vorticity_curl_fbo.deallocate()
        self._vorticity_force_fbo.deallocate()
        self._buoyancy_fbo.deallocate()
        self._expansion_fbo.deallocate()
        self._expansion_partials.deallocate()
        self._expansion_mean.deallocate()
        self._picture_fbo.deallocate()
        for scratch in (self._velocity_scratch, self._temperature_scratch, self._density_scratch):
            scratch.deallocate()

        self._advect_shader.deallocate()
        self._maccormack_shader.deallocate()
        self._divergence_shader.deallocate()
        self._expansion_mean_shader.deallocate()
        self._gradient_shader.deallocate()
        self._jacobi_pressure_shader.deallocate()
        self._jacobi_pressure_compute.deallocate()
        self._jacobi_diffusion_shader.deallocate()
        self._jacobi_diffusion_compute.deallocate()
        self._multigrid.deallocate()
        self._vorticity_curl_shader.deallocate()
        self._vorticity_force_shader.deallocate()
        self._buoyancy_shader.deallocate()
        self._add_boolean_shader.deallocate()
        for shader in self._input_shaders():
            shader.deallocate()
        self._gpu_timer.deallocate()
        self._guard.deallocate()

        self._allocated = False

    def _input_shaders(self) -> tuple:
        return (self._inject_velocity_shader, self._inject_density_shader, self._inject_scalar_shader, self._picture_shader)

    def _reload_shaders(self) -> None:
        """Hot-reload all shaders."""
        self._advect_shader.reload()
        self._maccormack_shader.reload()
        self._divergence_shader.reload()
        self._expansion_mean_shader.reload()
        self._gradient_shader.reload()
        self._jacobi_pressure_shader.reload()
        self._jacobi_pressure_compute.reload()
        self._jacobi_diffusion_shader.reload()
        self._jacobi_diffusion_compute.reload()
        self._multigrid.reload()
        self._vorticity_curl_shader.reload()
        self._vorticity_force_shader.reload()
        self._buoyancy_shader.reload()
        self._add_boolean_shader.reload()
        for shader in self._input_shaders():
            shader.reload()

    # ========== Update Pipeline ==========

    def reset(self) -> None:
        """Reset all simulation fields to zero."""
        if not self._allocated:
            return
        self._velocity_fbo.clear_all()
        self._density_fbo.clear_all()
        self._temperature_fbo.clear_all()
        self._temperature_clear = True
        self._pressure_fbo.clear_all()
        self._expansion_fbo.clear_all()
        self._picture_fbo.clear()

    def update(self, dt: float) -> None:
        """Run one step of the 2D fluid, in the 3D's order: the velocity carried, its viscosity, its confinement, the
        buoyancy, the projection, then the temperature and the density carried; `dt` is the tick's measured length (s)."""
        if not self._allocated:
            return

        self._reload_shaders()
        self._handle_deferred_actions()
        self._step_time(dt)
        self._gpu_timer.begin()

        self._advect_velocity()
        self._apply_viscosity()
        self._confine_vorticity()
        self._apply_buoyancy()
        self._enforce_incompressibility()
        self._advect_temperature()
        self._advect_density()

        self._follow_color_space()
        if self._oklab:
            self._picture_fbo.begin()
            self._picture_shader.use(self._density_fbo, True)
            self._picture_fbo.end()
        self._expansion_fresh = False      # the next tick's first injection clears it
        self._gpu_timer.end()
        self._guard.update(dt)

    # ========== Pipeline Steps ==========

    def _add_force_to_velocity(self, force: Texture, strength: float = 1.0) -> None:
        """Add `force` × `strength` to the velocity in place."""
        FlowUtil.add(self._velocity_fbo, force, strength)

    def _carry(self, field: SwapFbo, scratch: Fbo, velocity: Texture | None, obstacle: Texture, step: float,
               dissipation: float, advection: Advection, limiter: Limiter, evaporation: float, evaporation_knee: float,
               measure: int, guard: GuardSlot | None, wall_keep: float = 1.0) -> None:
        """Carry `field` one step along `velocity` (None: along itself) by `advection`: semi-Lagrangian into the field;
        or its step into `scratch` as the predictor, allocated at its first MacCormack step at the field's size, and
        the MacCormack correction by `limiter` into the field; on the result the fade, the evaporation read by
        `measure` (0 a density's alpha, 1 a vector's length, 2 |r|) and `guard`; the velocity's cells against a closed
        wall braked along it by `wall_keep` (1 for every other field)."""
        field.swap()
        source: Texture = field.back
        along: Texture = velocity if velocity is not None else source
        if advection != Advection.MACCORMACK:
            field.begin()
            self._advect_shader.use(source, along, obstacle, self._field_aspect, step, dissipation,
                                    has_obstacles=self._has_obstacles, evaporation=evaporation,
                                    evaporation_knee=evaporation_knee, measure=measure, guard=guard,
                                    closed=self._closed, wall_keep=wall_keep)
            field.end()
            return
        if not scratch.allocated or (scratch.width, scratch.height) != (field.width, field.height):
            scratch.deallocate()
            scratch.allocate(field.width, field.height, field.internal_format)
            self._apply_wall_modes()
        scratch.begin()
        self._advect_shader.use(source, along, obstacle, self._field_aspect, step, 1.0, has_obstacles=self._has_obstacles)
        scratch.end()
        field.begin()
        self._maccormack_shader.use(source, scratch, along, obstacle, self._field_aspect, step, dissipation,
                                    has_obstacles=self._has_obstacles, evaporation=evaporation,
                                    evaporation_knee=evaporation_knee, measure=measure, guard=guard,
                                    clamp=limiter == Limiter.CLAMP,
                                    closed=self._closed, wall_keep=wall_keep)
        field.end()

    def _advect_velocity(self) -> None:
        """The velocity carried one step along itself by `velocity.advection` (`velocity.advection_scale`), faded,
        evaporated and guarded."""
        velocity = self.settings.velocity
        self._carry(self._velocity_fbo, self._velocity_scratch, None, self._simulation_obstacle_fbo,
                    self._dt * velocity.advection_scale, self._calculate_dissipation(self._dt, velocity.fade_time),
                    velocity.advection, velocity.limiter, self._evaporation(self._dt, velocity.evaporation_time),
                    velocity.evaporation_knee, 1, self._guard_slot('velocity'), self._wall_keep)

    def _apply_viscosity(self) -> None:
        """The viscosity (widths² per second): Jacobi iterations of the implicit diffusion, `viscosity.iterations` at
        60 fps; while `viscosity.enabled` is on and the viscosity above 0."""
        viscosity = self.settings.viscosity
        if not viscosity.enabled or viscosity.viscosity <= 0.0:
            return

        viscosity_dt = viscosity.viscosity * self._dt       # widths² per second × s
        iterations = viscosity_iterations(viscosity.iterations, self._dt)

        if self._use_compute_diffusion:
            current: Texture = self._velocity_fbo
            result = self._jacobi_diffusion_compute.solve(
                current,
                self._velocity_fbo.back,
                self._simulation_obstacle_fbo,
                self._cell,
                viscosity_dt,
                total_iterations=iterations,
                iterations_per_dispatch=5,
                has_obstacles=self._has_obstacles,
                closed=self._closed,
                wall_keep=self._wall_keep
            )
            if result is not current:          # the solve ended in the back buffer: make it the front
                self._velocity_fbo.swap()
        else:
            for _ in range(iterations):
                self._velocity_fbo.swap()
                self._velocity_fbo.begin()
                self._jacobi_diffusion_shader.use(
                    self._velocity_fbo.back,
                    self._simulation_obstacle_fbo,
                    self._cell,
                    viscosity_dt,
                    has_obstacles=self._has_obstacles,
                    closed=self._closed,
                    wall_keep=self._wall_keep
                )
                self._velocity_fbo.end()

    def _confine_vorticity(self) -> None:
        """Vorticity confinement (Fedkiw 2001), each of the three (`vorticity.strength_1` to `_3` at `vorticity.radius_1`
        to `_3`, widths) keeping swirls of its own size alive: every curl in one pass, one channel each, their forces
        epsilon h (N × curl) × dt (h the cell's width) summed in one, added to the velocity once; while
        `vorticity.enabled` is on, each confinement while its strength is above 0."""
        vorticity = self.settings.vorticity
        running: list[tuple[float, float]] = [(strength, radius) for strength, radius in
                                              ((vorticity.strength_1, vorticity.radius_1),
                                               (vorticity.strength_2, vorticity.radius_2),
                                               (vorticity.strength_3, vorticity.radius_3))
                                              if vorticity.enabled and strength > 0.0 and radius > 0.0]
        if not running:
            return

        # a. Compute curl
        self._vorticity_curl_fbo.begin()
        self._vorticity_curl_shader.use(
            self._velocity_fbo,
            self._simulation_obstacle_fbo,
            self._cell,
            [radius for _, radius in running],
            has_obstacles=self._has_obstacles,
            closed=self._closed,
            wall_keep=self._wall_keep
        )
        self._vorticity_curl_fbo.end()

        # b. Compute confinement force × dt
        self._vorticity_force_fbo.begin()
        self._vorticity_force_shader.use(
            self._vorticity_curl_fbo,
            self._simulation_obstacle_fbo,
            self._cell,
            [strength * self._cell[0] * self._dt for strength, _ in running],
            has_obstacles=self._has_obstacles
        )
        self._vorticity_force_fbo.end()

        # c. Add force to velocity
        self._add_force_to_velocity(self._vorticity_force_fbo)

    @property
    def _temperature_runs(self) -> bool:
        """Whether the temperature is injected, carried and lifts: `temperature.enabled` and its `buoyancy` above 0."""
        temperature = self.settings.temperature
        return temperature.enabled and temperature.buoyancy > 0.0

    def _advect_temperature(self) -> None:
        """The temperature carried one step by the projected velocity, as the density is (its `advection`, `limiter`
        and `advection_scale`),
        faded and evaporated by its own; the buoyancy reads it the next step; while it does not run, cleared once and
        skipped."""
        if not self._temperature_runs:
            if not self._temperature_clear:     # cleared once; nothing adds to it while it does not run
                self._temperature_fbo.clear_all()
                self._temperature_clear = True
            return
        self._temperature_clear = False

        temperature = self.settings.temperature
        density = self.settings.density
        self._carry(self._temperature_fbo, self._temperature_scratch, self._velocity_fbo, self._simulation_obstacle_fbo,
                    self._dt * density.advection_scale, self._calculate_dissipation(self._dt, temperature.fade_time),
                    density.advection, density.limiter, self._evaporation(self._dt, temperature.evaporation_time),
                    temperature.evaporation_knee, 2, self._guard_slot('temperature'))

    def _apply_buoyancy(self) -> None:
        """The buoyancy, an acceleration f = gravity × (κρ − σT) (widths/s²) added × dt: the density's amount ρ sinks
        along gravity (`density.sink`), the warmth T rises against it (`temperature.buoyancy`, while the temperature
        runs). The pass runs while either is above 0 and `gravity` is not 0: one draw adds it straight to
        the velocity; the force itself is made only for its monitor."""
        shown: bool = self._buoyancy_shown
        self._buoyancy_shown = False
        lift: float = self.settings.temperature.buoyancy if self._temperature_runs else 0.0
        kappa: float = self.settings.density.sink
        g = self.settings.gravity
        gravity: tuple[float, float] = (g.x, g.y)
        if (lift <= 0.0 and kappa <= 0.0) or gravity == (0.0, 0.0):
            return

        velocity: SwapFbo = self._velocity_fbo
        velocity.swap()
        velocity.begin()
        self._buoyancy_shader.use(self._temperature_fbo, self._density_fbo, self._simulation_obstacle_fbo, lift, kappa,
                                  gravity, self._has_obstacles, velocity.back, self._dt)
        velocity.end()
        if shown:
            self._buoyancy_fbo.begin()
            self._buoyancy_shader.use(self._temperature_fbo, self._density_fbo, self._simulation_obstacle_fbo, lift,
                                      kappa, gravity, self._has_obstacles)
            self._buoyancy_fbo.end()

    def _show_buoyancy(self) -> None:
        """A monitor reads the buoyancy: the next step makes it."""
        self._buoyancy_shown = True

    def _enforce_incompressibility(self) -> None:
        """Pressure projection (divergence, less the expansion's rate × it → the solve by `pressure.solver` →
        gradient subtraction), in widths, on the walls `wall` sets; the screened equation (∇² − 1/reach²) p = div while
        `pressure.reach` is under 1 (`pressure_reach.screening`, the field's longest side its scale); with
        `pressure.enabled` off no pressure, all three skipped."""
        share: float = self.settings.pressure.reach
        if not self.settings.pressure.enabled or share <= 0.0:
            return
        closed: bool = self.settings.wall == WallMode.CLOSED
        screening: float = pressure_reach.screening(share, max(1.0, 1.0 / self._field_aspect))
        # a. Compute divergence, the expansion's made by the projection: its mean off where the walls are closed and the
        # solve needs the divergence to sum to 0, at reach 1
        rate: float = self._expansion_rate()
        mean: StorageBuffer | None = None
        if rate != 0.0 and closed and screening == 0.0:
            self._expansion_mean_shader.use(self._expansion_fbo, self._simulation_obstacle_fbo, self._has_obstacles,
                                            self._expansion_partials, self._expansion_mean)
            mean = self._expansion_mean
        self._divergence_fbo.begin()
        self._divergence_shader.use(
            self._velocity_fbo,
            self._simulation_obstacle_fbo,
            self._cell,
            has_obstacles=self._has_obstacles,
            expansion=self._expansion_fbo if rate != 0.0 else None,
            rate=rate,
            mean=mean,
            closed=closed,
            wall_keep=self._wall_keep
        )
        self._divergence_fbo.end()

        # b. Solve Poisson equation for pressure, by the solver picked
        pressure: PressureSettings = self.settings.pressure
        iterations = pressure.iterations     # the grid sets what it needs, not the step

        if pressure.solver == PressureSolver.MULTIGRID:
            self._multigrid.solve(self._pressure_fbo, self._divergence_fbo, self._simulation_obstacle_fbo, self._cell,
                                  pressure.cycles, self._has_obstacles, closed, screening)
        elif pressure.solver == PressureSolver.SOR:
            self._multigrid.sor(self._pressure_fbo, self._divergence_fbo, self._simulation_obstacle_fbo, self._cell,
                                iterations, pressure.relaxation, self._has_obstacles, closed, screening)
        elif self._use_compute_pressure:
            current = self._pressure_fbo
            result = self._jacobi_pressure_compute.solve(
                current,
                self._pressure_fbo.back,
                self._divergence_fbo,
                self._simulation_obstacle_fbo,
                self._cell,
                total_iterations=iterations,
                iterations_per_dispatch=5,
                has_obstacles=self._has_obstacles,
                closed=closed,
                screening=screening
            )
            if result is not current:          # the solve ended in the back buffer: make it the front
                self._pressure_fbo.swap()
        else:
            for _ in range(iterations):
                self._pressure_fbo.swap()
                self._pressure_fbo.begin()
                self._jacobi_pressure_shader.use(
                    self._pressure_fbo.back,
                    self._divergence_fbo,
                    self._simulation_obstacle_fbo,
                    self._cell,
                    has_obstacles=self._has_obstacles,
                    closed=closed,
                    screening=screening
                )
                self._pressure_fbo.end()

        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)       # the compute solves' writes, before the gradient reads them

        # c. Subtract pressure gradient from velocity
        self._velocity_fbo.swap()
        self._velocity_fbo.begin()
        self._gradient_shader.use(
            self._velocity_fbo.back,
            self._pressure_fbo,
            self._simulation_obstacle_fbo,
            self._cell,
            has_obstacles=self._has_obstacles,
            closed=closed
        )
        self._velocity_fbo.end()

    def _advect_density(self) -> None:
        """The density carried one step by the projected velocity by `density.advection` (`density.advection_scale`),
        faded, evaporated and guarded."""
        density = self.settings.density
        self._carry(self._density_fbo, self._density_scratch, self._velocity_fbo, self._density_obstacle_fbo,
                    self._dt * density.advection_scale, self._calculate_dissipation(self._dt, density.fade_time),
                    density.advection, density.limiter, self._evaporation(self._dt, density.evaporation_time),
                    density.evaporation_knee, 0, self._guard_slot('density'))

    @property
    def _closed(self) -> bool:
        """Whether the walls are closed (`wall`), as every pass reading beyond the field takes them."""
        return self.settings.wall == WallMode.CLOSED

    @property
    def _wall_keep(self) -> float:
        """The velocity's share along a closed wall kept this step: 0.01 over `slip_time` seconds, 1 at 0 (free-slip)."""
        slip_time: float = self.settings.slip_time
        return pow(0.01, self._dt / slip_time) if slip_time > 0.0 else 1.0

    def _guard_slot(self, field: str) -> GuardSlot | None:
        """The field's slot in the fluid's guard, None while `guard` is off."""
        return self._guard.slot(field) if self.settings.guard else None

    # ========== Deferred ==========

    def _apply_wall_modes(self) -> None:
        """Apply `wall` to the velocity, the density and the temperature, and their predictors, read beyond the walls
        (GL thread only): closed, the edge; open, still water, 0. The pressure's walls are the solvers' ghosts."""
        wall = _WALL_GL[self.settings.wall]
        for fbo in (self._velocity_fbo, self._density_fbo, self._temperature_fbo, self._velocity_scratch, self._density_scratch,
                    self._temperature_scratch):
            if fbo.allocated:
                fbo.set_wrap(*wall)

    def _handle_deferred_actions(self) -> None:
        """Process reset and reallocation requests queued from the UI thread."""
        if self._reallocate_pending:
            self._reallocate_pending = False
            old_w, old_h = self._simulation_width, self._simulation_height
            self._update_simulation_dimensions()
            if self._simulation_width != old_w or self._simulation_height != old_h:
                self._allocate_simulation_fields()

        if self._wall_dirty:
            self._wall_dirty = False
            self._apply_wall_modes()

    def _request_reallocate(self) -> None:
        """Thread-safe reallocation request — deferred to next update() on the GL thread."""
        self._reallocate_pending = True

    # ========== Internal helpers ==========

    @property
    def dt(self) -> float:
        """The time step of the last update (s)."""
        return self._dt

    def _step_time(self, dt: float) -> None:
        time = self.settings.time
        # the fluid's own time: every pass steps by it, the injections' sources and the drag included, so at
        # time_scale 0.5 the whole fluid runs at half speed and keeps its shapes
        self._dt = self._time_step.update(time.time_step, dt, time.fixed_fps, time.fps_smoothing) * time.time_scale

    @staticmethod
    def _align8(v: int) -> int:
        """Round up to the nearest multiple of 8."""
        return (v + 7) & ~7

    @staticmethod
    def _calculate_dissipation(delta_time: float, decay_time: float) -> float:
        """The fade's multiplier this step, pow(0.01, dt / decay_time): the field at 1% after decay_time seconds;
        1 at 0, off."""
        if decay_time <= 0.0:
            return 1.0
        return pow(0.01, delta_time / decay_time)

    @staticmethod
    def _evaporation(delta_time: float, evaporation_time: float) -> float:
        """The saturating loss's share this step, dt / evaporation_time: a full value well above the knee is gone in
        evaporation_time; 0 off."""
        return delta_time / evaporation_time if evaporation_time > 0.0 else 0.0

    # ========== Input Methods: what the bridge gives, as the 3D fluid takes it ==========

    def drag_velocity(self, texture: Texture, weight: float = 1.0) -> None:
        """The boundary: drag the velocity toward the body's (x, y on the image plane, widths/s) × input.velocity_gain,
        over input.velocity_drag, where the body moves (its speed above DRAG_FLOOR), never past it."""
        if not self._allocated:
            return
        input: InputSettings = self.settings.input
        self._velocity_fbo.swap()
        self._velocity_fbo.begin()
        self._inject_velocity_shader.drag(self._velocity_fbo.back, texture, input.velocity_gain,
                                          drag_share(self._dt, input.velocity_drag), weight, DRAG_FLOOR)
        self._velocity_fbo.end()

    def add_acceleration(self, texture: Texture, weight: float = 1.0) -> None:
        """The body force: the acceleration (widths/s²) × input.acceleration_gain × dt added."""
        if not self._allocated:
            return
        self._velocity_fbo.swap()
        self._velocity_fbo.begin()
        self._inject_velocity_shader.add(self._velocity_fbo.back, texture, weight * self.settings.input.acceleration_gain * self._dt)
        self._velocity_fbo.end()

    def add_density(self, amount: Texture, color: Texture, weight: float = 1.0) -> None:
        """The boundary's dye, the bridge's `amount` (red) times input.density_gain times its `color` (RGBA,
        premultiplied by its alpha, the coverage), by input.density_blend: over the water's dye by input.density_drag's
        share, the amount capped at 1; added × dt; or the water drawn toward it, at most fully dyed, over
        input.density_drag; where the water holds more than the body, down over input.density_release (0 = never
        down)."""
        if not self._allocated:
            return
        input: InputSettings = self.settings.input
        blend: DensityBlend = input.density_blend
        strength: float = (self._dt if blend == DensityBlend.ADD else drag_share(self._dt, input.density_drag)) * weight
        release: float = drag_share(self._dt, input.density_release) * weight if input.density_release > 0.0 else 0.0
        self._follow_color_space()
        self._density_fbo.swap()
        self._density_fbo.begin()
        self._inject_density_shader.use(self._density_fbo.back, amount, color, strength, release, blend, input.density_gain,
                                        self._oklab)
        self._density_fbo.end()

    def add_temperature(self, texture: Texture, weight: float = 1.0) -> None:
        """The boundary's warmth: the bridge's presence (0 to 1) times input.temperature_gain, at most ±1, the water
        dragged toward it over input.temperature_drag where the body is warmer (or colder) than the water; never back
        toward a fainter body, as the density's, cooled by the fade alone."""
        if not self._allocated or not self._temperature_runs:      # nothing reads it
            return
        input: InputSettings = self.settings.input
        self._temperature_fbo.swap()
        self._temperature_fbo.begin()
        self._inject_scalar_shader.use(self._temperature_fbo.back, texture, ScalarInjection.FILL,
                                       drag_share(self._dt, input.temperature_drag) * weight, input.temperature_gain)
        self._temperature_fbo.end()

    def add_expansion(self, texture: Texture, weight: float = 1.0) -> None:
        """The expansion (a presence, 0 to 1) added, summed over this tick's injections; the projection makes the
        divergence expansion.rate (1/s at full presence) times it, while `expansion.enabled` is on and the rate not 0.
        The first injection of a tick clears the last tick's."""
        if not self._allocated or self._expansion_rate() == 0.0:
            return
        if not self._expansion_fresh:
            self._expansion_fbo.clear_all()
            self._expansion_fresh = True
        self._expansion_fbo.swap()
        self._expansion_fbo.begin()
        self._inject_scalar_shader.use(self._expansion_fbo.back, texture, ScalarInjection.ADD, weight)
        self._expansion_fbo.end()

    def _expansion_rate(self) -> float:
        """The expansion's rate while it runs, else 0."""
        expansion = self.settings.expansion
        return expansion.rate if expansion.enabled else 0.0

    def _follow_color_space(self) -> None:
        """The density cleared when its color space changes: its values read wrong in the other."""
        oklab: bool = self.settings.density.color_space == ColorSpace.OKLAB
        if oklab != self._oklab:
            self._oklab = oklab
            self._density_fbo.clear_all()

    # ========== Setting a field, or a region of it ==========

    def set_velocity(self, texture: Texture, strength: float = 1.0) -> None:
        """Set velocity field."""
        if not self._allocated:
            return
        FlowUtil.set(self._velocity_fbo, texture, strength)

    def set_velocity_region(self, texture: Texture, x: float, y: float, w: float, h: float, strength: float = 1.0) -> None:
        """Set velocity in a specific region."""
        if not self._allocated:
            return
        FlowUtil.set_region(self._velocity_fbo, texture, x, y, w, h, strength)

    def add_velocity_region(self, texture: Texture, x: float, y: float, w: float, h: float, strength: float = 1.0) -> None:
        """Add velocity to a specific region."""
        if not self._allocated:
            return
        FlowUtil.add_region(self._velocity_fbo, texture, x, y, w, h, strength)

    # ----- Density -----
    def set_density(self, texture: Texture, strength: float = 1.0) -> None:
        """Set density field."""
        if not self._allocated:
            return
        FlowUtil.set(self._density_fbo, texture, strength)

    def set_density_region(self, texture: Texture, x: float, y: float, w: float, h: float, strength: float = 1.0) -> None:
        """Set density in a specific region."""
        if not self._allocated:
            return
        FlowUtil.set_region(self._density_fbo, texture, x, y, w, h, strength)

    def set_density_channel(self, texture: Texture, channel: int) -> None:
        """Set single-channel texture to one of the density channels."""
        if not self._allocated:
            return
        FlowUtil.set_channel(self._density_fbo, texture, channel)

    def set_density_channel_region(self, texture: Texture, channel: int, x: float, y: float, w: float, h: float, strength: float = 1.0) -> None:
        """Set density channel in a specific region."""
        if not self._allocated:
            return
        FlowUtil.set_channel_region(self._density_fbo, texture, channel, x, y, w, h, strength)

    def add_density_region(self, texture: Texture, x: float, y: float, w: float, h: float, strength: float = 1.0) -> None:
        """Add density to a specific region."""
        if not self._allocated:
            return
        FlowUtil.add_region(self._density_fbo, texture, x, y, w, h, strength)

    # ----- Temperature -----
    def set_temperature(self, texture: Texture, strength: float = 1.0) -> None:
        """Set temperature field."""
        if not self._allocated:
            return
        FlowUtil.set(self._temperature_fbo, texture, strength)

    def set_temperature_region(self, texture: Texture, x: float, y: float, w: float, h: float, strength: float = 1.0) -> None:
        """Set temperature in a specific region."""
        if not self._allocated:
            return
        FlowUtil.set_region(self._temperature_fbo, texture, x, y, w, h, strength)

    def add_temperature_region(self, texture: Texture, x: float, y: float, w: float, h: float, strength: float = 1.0) -> None:
        """Add temperature to a specific region."""
        if not self._allocated:
            return
        FlowUtil.add_region(self._temperature_fbo, texture, x, y, w, h, strength)

    # ----- Pressure -----
    def set_pressure(self, texture: Texture, strength: float = 1.0) -> None:
        """Set pressure field."""
        if not self._allocated:
            return
        FlowUtil.set(self._pressure_fbo, texture, strength)

    def set_pressure_region(self, texture: Texture, x: float, y: float, w: float, h: float, strength: float = 1.0) -> None:
        """Set pressure in a specific region."""
        if not self._allocated:
            return
        FlowUtil.set_region(self._pressure_fbo, texture, x, y, w, h, strength)

    def add_pressure(self, texture: Texture, strength: float = 1.0) -> None:
        """Add to pressure field."""
        if not self._allocated:
            return
        FlowUtil.add(self._pressure_fbo, texture, strength)

    def add_pressure_region(self, texture: Texture, x: float, y: float, w: float, h: float, strength: float = 1.0) -> None:
        """Add pressure to a specific region."""
        if not self._allocated:
            return
        FlowUtil.add_region(self._pressure_fbo, texture, x, y, w, h, strength)

    # ----- Obstacles -----
    def set_obstacle(self, texture: Texture) -> None:
        """Replace obstacle mask with texture (boolean OR with empty field).

        Updates both sim-resolution and density-resolution obstacle buffers.

        Args:
            texture: Obstacle mask (any channel > 0 = obstacle)
        """
        if not self._allocated:
            return
        FlowUtil.zero(self._simulation_obstacle_fbo)
        self._simulation_obstacle_fbo.swap()
        self._simulation_obstacle_fbo.begin()
        self._add_boolean_shader.use(self._simulation_obstacle_fbo.back, texture)
        self._simulation_obstacle_fbo.end()

        FlowUtil.zero(self._density_obstacle_fbo)
        self._density_obstacle_fbo.swap()
        self._density_obstacle_fbo.begin()
        self._add_boolean_shader.use(self._density_obstacle_fbo.back, texture)
        self._density_obstacle_fbo.end()

        self._has_obstacles = True

    def add_obstacle(self, texture: Texture) -> None:
        """Add to obstacle mask (boolean OR).

        Updates both sim-resolution and density-resolution obstacle buffers.

        Args:
            texture: Obstacle mask to add
        """
        if not self._allocated:
            return
        self._simulation_obstacle_fbo.swap()
        self._simulation_obstacle_fbo.begin()
        self._add_boolean_shader.use(self._simulation_obstacle_fbo.back, texture)
        self._simulation_obstacle_fbo.end()

        self._density_obstacle_fbo.swap()
        self._density_obstacle_fbo.begin()
        self._add_boolean_shader.use(self._density_obstacle_fbo.back, texture)
        self._density_obstacle_fbo.end()

        self._has_obstacles = True

    def clear_obstacles(self) -> None:
        """Clear all obstacles."""
        if not self._allocated:
            return
        FlowUtil.zero(self._simulation_obstacle_fbo)
        FlowUtil.zero(self._density_obstacle_fbo)
        self._has_obstacles = False

    # ========== Properties ==========

    @property
    def allocated(self) -> bool:
        return self._allocated

    @property
    def velocity(self) -> Texture:
        """RG16F velocity field."""
        return self._velocity_fbo

    @property
    def density(self) -> Texture:
        """RGBA16F density, premultiplied: its color times its amount (RGB or Oklab), and its amount."""
        return self._density_fbo

    @property
    def picture(self) -> Texture:
        """The density as RGB, premultiplied by its amount: the density itself held as RGB, turned from Oklab else."""
        return self._picture_fbo if self._oklab else self._density_fbo

    @property
    def temperature(self) -> Texture:
        """R16F temperature field."""
        return self._temperature_fbo

    @property
    def pressure(self) -> Texture:
        """R32F pressure field."""
        return self._pressure_fbo

    @property
    def divergence(self) -> Texture:
        """R16F divergence field (intermediate)."""
        return self._divergence_fbo

    @property
    def vorticity_curl(self) -> Texture:
        """R16F vorticity curl field."""
        return self._vorticity_curl_fbo

    @property
    def buoyancy(self) -> Texture:
        """RG16F buoyancy, an acceleration (widths/s²)."""
        return self._buoyancy_fbo

    @property
    def obstacle(self) -> Texture:
        """R8 obstacle mask."""
        return self._density_obstacle_fbo

    @property
    def ports(self) -> dict[str, Port]:
        """Its fields, each in the order a step reaches it."""
        return {'obstacle': Port(self._density_obstacle_fbo, TextureKind.SCALAR),
                'velocity': Port(self._velocity_fbo, TextureKind.VELOCITY),
                'vorticity': Port(self._vorticity_curl_fbo, TextureKind.SCALAR),
                'vorticity_force': Port(self._vorticity_force_fbo, TextureKind.VELOCITY),
                'buoyancy': Port(self._buoyancy_fbo, TextureKind.ACCELERATION, prepare=self._show_buoyancy),
                'expansion': Port(self._expansion_fbo, TextureKind.SCALAR),
                'divergence': Port(self._divergence_fbo, TextureKind.SCALAR),
                'pressure': Port(self._pressure_fbo, TextureKind.SCALAR),
                'temperature': Port(self._temperature_fbo, TextureKind.SCALAR),
                'density': Port(self._density_fbo, TextureKind.IMAGE,
                                color_space=ColorSpace.OKLAB if self._oklab else ColorSpace.RGB)}

    @property
    def sim_width(self) -> int:
        """Current simulation resolution width (aligned to 8)."""
        return self._simulation_width

    @property
    def sim_height(self) -> int:
        """Current simulation resolution height (aligned to 8)."""
        return self._simulation_height
