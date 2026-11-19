"""Fluid2D - 2D Navier-Stokes fluid simulation.

All-fragment pipeline operating on GL_TEXTURE_2D FBOs with ping-pong
buffers for iterative solvers.

Inputs, before the step, as the 3D fluid takes them: the boundary's velocity dragged toward, the body force added
× dt, the density over, added or filled with its color (RGB or Oklab), the temperature dragged toward, the expansion
summed for the projection. The velocity is in widths/s; the advection and the projection work in widths.

Pipeline:
    1. Dampen velocity (magnitude clamping)
    2. Advect velocity (self-advection)
    3. Apply viscosity (Jacobi diffusion solver)
    4. Confine vorticity (curl + confinement force)
    5. Advect temperature
    6. Apply buoyancy force
    7. Advect pressure (optional, non-physical)
    8. Enforce incompressibility (divergence, less the expansion's rate × it -> Jacobi solve -> gradient subtraction)
    9. Advect density (semi-Lagrangian)
   10. Dampen density (magnitude clamping)

Boundary conditions via per-field wrap modes:
    Velocity/density:  GL_CLAMP_TO_BORDER(0,0,0,0)  -> no-slip / Dirichlet
    Pressure/temp:     GL_CLAMP_TO_EDGE              -> zero-gradient / Neumann
    Obstacle:          GL_CLAMP_TO_BORDER(1,1,1,1)   -> OOB = obstacle
"""
import math

from OpenGL.GL import GL_CLAMP_TO_BORDER, GL_CLAMP_TO_EDGE, GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR, GL_R16F, GL_R8, GL_RG16F, GL_RGBA16F, \
    GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER, glBindTexture, glGenerateMipmap, glTexParameteri

from modules.gl import Texture, SwapFbo, Fbo, GpuTimer, Profiler
from .. import FlowUtil, Port, TextureKind
from ..density_blend import DensityBlend
from ..drag import DRAG_FLOOR, drag_share
from ..port import ColorSpace
from ..time_step import REFERENCE_STEP, TimeStep, viscosity_iterations
from ..shaders import AddBoolean
from ..wall_mode import WallMode
from .fluid_settings import Fluid2DSettings, InputSettings, VelocitySettings, DensitySettings, TemperatureSettings, PressureSettings
from .shaders import (
    Advect, Divergence, Gradient,
    JacobiPressure, JacobiPressureCompute, JacobiDiffusion, JacobiDiffusionCompute,
    VorticityCurl, VorticityForce, Buoyancy,
    InjectVelocity2D, InjectDensity2D, InjectScalar2D, ScalarInjection, DensityToRgb2D
)

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
        pressure:             SwapFbo R16F
        simulation_obstacle:  SwapFbo R8        (sim resolution)
        density_obstacle:     SwapFbo R8        (density resolution)

    Intermediate (single-buffer):
        divergence:       Fbo R16F
        vorticity_curl:   Fbo R16F
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
        self._aspect: float = 1.0          # the simulation grid's width over its height, for the viscosity and the vorticity
        self._field_aspect: float = 1.0    # the field's width over its height, the density's: one width is this of the height
        self._cell: tuple[float, float] = (1.0, 1.0)   # a simulation cell's width and height, widths

        self._dt: float = REFERENCE_STEP
        self._time_step: TimeStep = TimeStep()

        self._has_obstacles: bool = False

        self._allocated: bool = False

        self._reallocate_pending: bool = False
        self._wall_dirty: bool = False

        # ---- Simulation fields (SwapFbo for ping-pong) ----
        # Velocity always uses Dirichlet (v=0 at walls); others configurable via _apply_wall_modes().
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
        self._expansion_fbo: SwapFbo = SwapFbo()     # the expansion over this tick's injections, mipmapped for its mean
        self._expansion_fresh: bool = False          # this tick's first injection has cleared it
        self._picture_fbo: Fbo = Fbo()               # the density as RGB, premultiplied, where it is held as Oklab
        self._oklab: bool = self.settings.density.color_space == ColorSpace.OKLAB
        # Curl is sampled at neighbor offsets by VorticityForce — border must return zero
        self._vorticity_curl_fbo: Fbo = Fbo(wrap=GL_CLAMP_TO_BORDER, border_color=(0.0, 0.0, 0.0, 0.0))
        self._vorticity_force_fbo: Fbo = Fbo()
        self._buoyancy_fbo: Fbo = Fbo()

        # Shaders
        self._advect_shader: Advect = Advect()
        self._divergence_shader: Divergence = Divergence()
        self._gradient_shader: Gradient = Gradient()
        self._jacobi_pressure_shader: JacobiPressure = JacobiPressure()
        self._jacobi_pressure_compute: JacobiPressureCompute = JacobiPressureCompute()
        self._jacobi_diffusion_shader: JacobiDiffusion = JacobiDiffusion()
        self._jacobi_diffusion_compute: JacobiDiffusionCompute = JacobiDiffusionCompute()

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

        # ---- GPU profiling ----
        self._profiler: Profiler = Profiler(
            lambda: f"Fluid2D {self._simulation_width}x{self._simulation_height}",
            names=["dampen_vel", "advect_vel", "viscosity", "vorticity",
                   "advect_temp", "buoyancy", "advect_pres", "incompress",
                   "advect_den", "dampen_den"],
        )

        # Bind settings actions
        self.settings.bind(Fluid2DSettings.width, lambda _: self._request_reallocate())
        self.settings.bind(Fluid2DSettings.height, lambda _: self._request_reallocate())
        self.settings.bind(Fluid2DSettings.profile, lambda v: setattr(self._profiler, 'enabled', v))
        mark_wall = lambda _: setattr(self, '_wall_dirty', True)
        self.settings.density.bind(DensitySettings.wall, mark_wall)
        self.settings.temperature.bind(TemperatureSettings.wall, mark_wall)
        self.settings.pressure.bind(PressureSettings.wall, mark_wall)

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
        self._divergence_shader.allocate()
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

        self._allocated = True

    def _update_simulation_dimensions(self) -> None:
        """Recompute simulation dimensions from current settings."""
        self._simulation_width = self._align32(self.settings.width)
        self._simulation_height = self._align32(self.settings.height)
        self._aspect = self._simulation_width / self._simulation_height if self._simulation_height > 0 else 1.0
        self._grid_scale = self._simulation_width / self._density_width if self._density_width > 0 else 1.0
        self._field_aspect = self._density_width / self._density_height if self._density_height > 0 else 1.0
        # the field is one width across and 1 / field_aspect widths high, the grid's cells over it
        self._cell = (1.0 / max(self._simulation_width, 1), 1.0 / (self._field_aspect * max(self._simulation_height, 1)))

    def _allocate_density_fields(self) -> None:
        """(Re)allocate density-resolution FBOs (full output resolution)."""
        self._density_fbo.allocate(self._density_width, self._density_height, GL_RGBA16F)
        self._picture_fbo.allocate(self._density_width, self._density_height, GL_RGBA16F)
        self._density_obstacle_fbo.allocate(self._density_width, self._density_height, GL_R8)

    def _allocate_simulation_fields(self) -> None:
        """(Re)allocate simulation-resolution FBOs."""
        sim_w = self._simulation_width
        sim_h = self._simulation_height

        # Primary simulation fields
        self._velocity_fbo.allocate(sim_w, sim_h, GL_RG16F)
        self._temperature_fbo.allocate(sim_w, sim_h, GL_R16F)
        self._pressure_fbo.allocate(sim_w, sim_h, GL_R16F)
        self._simulation_obstacle_fbo.allocate(sim_w, sim_h, GL_R8)
        FlowUtil.blit(self._simulation_obstacle_fbo, self._density_obstacle_fbo)

        # Intermediate FBOs
        self._divergence_fbo.allocate(sim_w, sim_h, GL_R16F)
        self._vorticity_curl_fbo.allocate(sim_w, sim_h, GL_R16F)
        self._vorticity_force_fbo.allocate(sim_w, sim_h, GL_RG16F)
        self._buoyancy_fbo.allocate(sim_w, sim_h, GL_RG16F)
        self._expansion_fbo.allocate(sim_w, sim_h, GL_R16F)
        for _ in range(2):      # both buffers mipmapped: its mips, its mean
            glBindTexture(GL_TEXTURE_2D, self._expansion_fbo.tex_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glGenerateMipmap(GL_TEXTURE_2D)
            self._expansion_fbo.swap()
        glBindTexture(GL_TEXTURE_2D, 0)
        self._expansion_fbo.clear_all()

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
        self._picture_fbo.deallocate()

        self._advect_shader.deallocate()
        self._divergence_shader.deallocate()
        self._gradient_shader.deallocate()
        self._jacobi_pressure_shader.deallocate()
        self._jacobi_pressure_compute.deallocate()
        self._jacobi_diffusion_shader.deallocate()
        self._jacobi_diffusion_compute.deallocate()
        self._vorticity_curl_shader.deallocate()
        self._vorticity_force_shader.deallocate()
        self._buoyancy_shader.deallocate()
        self._add_boolean_shader.deallocate()
        for shader in self._input_shaders():
            shader.deallocate()
        self._gpu_timer.deallocate()

        self._allocated = False

    def _input_shaders(self) -> tuple:
        return (self._inject_velocity_shader, self._inject_density_shader, self._inject_scalar_shader, self._picture_shader)

    def _reload_shaders(self) -> None:
        """Hot-reload all shaders."""
        self._advect_shader.reload()
        self._divergence_shader.reload()
        self._gradient_shader.reload()
        self._jacobi_pressure_shader.reload()
        self._jacobi_pressure_compute.reload()
        self._jacobi_diffusion_shader.reload()
        self._jacobi_diffusion_compute.reload()
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
        self._pressure_fbo.clear_all()
        self._expansion_fbo.clear_all()
        self._picture_fbo.clear()

    def update(self, dt: float) -> None:
        """Run one frame of the 2D fluid simulation pipeline; `dt` is the tick's measured length (s)."""
        if not self._allocated:
            return

        self._reload_shaders()
        self._handle_deferred_actions()
        self._gpu_timer.begin()

        # Per-frame state
        self._step_time(dt)

        # Dampen velocity (clean input for all steps)
        vel: VelocitySettings = self.settings.velocity
        self._profiler.begin("dampen_vel")
        self._dampen(self._velocity_fbo, vel.dampen_threshold, vel.dampen_time, self._dt, include_alpha=False)
        self._profiler.end("dampen_vel")

        # Simulation steps
        self._profiler.begin("advect_vel")
        self._advect_velocity()
        self._profiler.end("advect_vel")

        self._profiler.begin("viscosity")
        self._apply_viscosity()
        self._profiler.end("viscosity")

        self._profiler.begin("vorticity")
        self._confine_vorticity()
        self._profiler.end("vorticity")

        self._profiler.begin("advect_temp")
        self._advect_temperature()
        self._profiler.end("advect_temp")

        self._profiler.begin("buoyancy")
        self._apply_buoyancy()
        self._profiler.end("buoyancy")

        self._profiler.begin("advect_pres")
        self._advect_pressure()
        self._profiler.end("advect_pres")

        self._profiler.begin("incompress")
        self._enforce_incompressibility()
        self._profiler.end("incompress")

        self._profiler.begin("advect_den")
        self._advect_density()
        self._profiler.end("advect_den")

        # Dampen density (clean output)
        den: DensitySettings = self.settings.density
        self._profiler.begin("dampen_den")
        self._dampen(self._density_fbo, den.dampen_threshold, den.dampen_time, self._dt, include_alpha=True)
        self._profiler.end("dampen_den")

        self._follow_color_space()
        if self._oklab:
            self._picture_fbo.begin()
            self._picture_shader.use(self._density_fbo, True)
            self._picture_fbo.end()
        self._expansion_fresh = False      # the next tick's first injection clears it
        self._gpu_timer.end()

        self._profiler.report()

    # ========== Pipeline Steps ==========

    def _add_force_to_velocity(self, force: Texture, strength: float = 1.0) -> None:
        """Add force to velocity in-place (no gain or dt scaling)."""
        FlowUtil.add(self._velocity_fbo, force, strength)

    def _advect_velocity(self) -> None:
        """Self-advect & dissipate velocity field."""
        advect_step = self._dt * self.settings.velocity.self_advection
        dissipation = self._calculate_dissipation(self._dt, self.settings.velocity.fade_time)

        self._velocity_fbo.swap()
        self._velocity_fbo.begin()
        self._advect_shader.use(
            self._velocity_fbo.back,   # Source velocity (self-advection)
            self._velocity_fbo.back,   # Velocity
            self._simulation_obstacle_fbo,     # Obstacles
            self._field_aspect,
            advect_step,
            dissipation,
            has_obstacles=self._has_obstacles
        )
        self._velocity_fbo.end()

    def _apply_viscosity(self) -> None:
        """Diffuse velocity via Jacobi viscosity solver."""
        if self.settings.velocity.viscosity <= 0.0:
            return

        viscosity_dt = self.settings.velocity.viscosity * (self._grid_scale ** 2) * self._dt
        iterations = viscosity_iterations(self.settings.velocity.viscosity_iter, self._dt)

        if self._use_compute_diffusion:
            current: Texture = self._velocity_fbo
            result = self._jacobi_diffusion_compute.solve(
                current,
                self._velocity_fbo.back,
                self._simulation_obstacle_fbo,
                self._grid_scale,
                self._aspect,
                viscosity_dt,
                total_iterations=iterations,
                iterations_per_dispatch=5,
                has_obstacles=self._has_obstacles
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
                    self._grid_scale,
                    self._aspect,
                    viscosity_dt,
                    has_obstacles=self._has_obstacles
                )
                self._velocity_fbo.end()

    def _confine_vorticity(self) -> None:
        """Vorticity confinement (curl → force → add to velocity)."""
        if self.settings.velocity.vorticity <= 0.0 or self.settings.velocity.vorticity_radius <= 0.0:
            return

        vorticity_radius = self.settings.velocity.vorticity_radius * self._grid_scale
        vorticity_force = self.settings.velocity.vorticity * self._dt

        # a. Compute curl
        self._vorticity_curl_fbo.begin()
        self._vorticity_curl_shader.use(
            self._velocity_fbo,
            self._simulation_obstacle_fbo,
            self._grid_scale,
            self._aspect,
            vorticity_radius,
            has_obstacles=self._has_obstacles
        )
        self._vorticity_curl_fbo.end()

        # b. Compute confinement force
        self._vorticity_force_fbo.begin()
        self._vorticity_force_shader.use(
            self._vorticity_curl_fbo,
            self._simulation_obstacle_fbo,
            self._grid_scale,
            self._aspect,
            vorticity_force,
            has_obstacles=self._has_obstacles
        )
        self._vorticity_force_fbo.end()

        # c. Add force to velocity
        self._add_force_to_velocity(self._vorticity_force_fbo)

    def _advect_temperature(self) -> None:
        """Advect & dissipate the temperature scalar field."""
        if self.settings.temperature.buoyancy == 0.0:
            self._temperature_fbo.clear_all()
            return

        advect_step = self._dt * self.settings.speed
        dissipation = self._calculate_dissipation(self._dt, self.settings.temperature.fade_time)

        self._temperature_fbo.swap()
        self._temperature_fbo.begin()
        self._advect_shader.use(
            self._temperature_fbo.back,
            self._velocity_fbo,
            self._simulation_obstacle_fbo,
            self._field_aspect,
            advect_step,
            dissipation,
            has_obstacles=self._has_obstacles
        )
        self._temperature_fbo.end()

    def _apply_buoyancy(self) -> None:
        """Apply buoyancy force to velocity: F = σ(T − T_ambient) − κρ."""
        if self.settings.temperature.buoyancy == 0.0:
            return

        sigma = self._dt * self._grid_scale * self.settings.temperature.buoyancy
        kappa = self._dt * self._grid_scale * self.settings.temperature.sink

        self._buoyancy_fbo.begin()
        self._buoyancy_shader.use(
            self._velocity_fbo,
            self._temperature_fbo,
            self._density_fbo,
            self._simulation_obstacle_fbo,
            sigma,
            kappa,
            self.settings.temperature.ambient,
            has_obstacles=self._has_obstacles
        )
        self._buoyancy_fbo.end()

        self._add_force_to_velocity(self._buoyancy_fbo)

    def _advect_pressure(self) -> None:
        """Advect & dissipate pressure (optional, non-physical)."""
        if self.settings.pressure.speed <= 0.0:
            return

        advect_step = self._dt * self.settings.pressure.speed
        dissipation = self._calculate_dissipation(self._dt, self.settings.pressure.fade_time)

        self._pressure_fbo.swap()
        self._pressure_fbo.begin()
        self._advect_shader.use(
            self._pressure_fbo.back,
            self._velocity_fbo,
            self._simulation_obstacle_fbo,
            self._field_aspect,
            advect_step,
            dissipation,
            has_obstacles=self._has_obstacles
        )
        self._pressure_fbo.end()

    def _enforce_incompressibility(self) -> None:
        """Pressure projection (divergence, less the expansion's rate × it → Jacobi solve → gradient subtraction), in
        widths."""
        # a. Compute divergence, the expansion's made by the projection (its mean off where the walls are closed)
        rate: float = self._expansion_rate()
        mean_lod: float | None = None
        if rate != 0.0:
            glBindTexture(GL_TEXTURE_2D, self._expansion_fbo.tex_id)
            glGenerateMipmap(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, 0)
            if self.settings.pressure.wall == WallMode.CLOSED:
                mean_lod = float(math.floor(math.log2(max(self._simulation_width, self._simulation_height))))
        self._divergence_fbo.begin()
        self._divergence_shader.use(
            self._velocity_fbo,
            self._simulation_obstacle_fbo,
            self._cell,
            has_obstacles=self._has_obstacles,
            expansion=self._expansion_fbo if rate != 0.0 else None,
            rate=rate,
            mean_lod=mean_lod
        )
        self._divergence_fbo.end()

        # b. Solve Poisson equation for pressure
        iterations = self.settings.pressure.iterations     # the grid sets what it needs, not the step

        if self._use_compute_pressure:
            current = self._pressure_fbo
            result = self._jacobi_pressure_compute.solve(
                current,
                self._pressure_fbo.back,
                self._divergence_fbo,
                self._simulation_obstacle_fbo,
                self._cell,
                total_iterations=iterations,
                iterations_per_dispatch=5,
                has_obstacles=self._has_obstacles
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
                    has_obstacles=self._has_obstacles
                )
                self._pressure_fbo.end()

        # c. Subtract pressure gradient from velocity
        self._velocity_fbo.swap()
        self._velocity_fbo.begin()
        self._gradient_shader.use(
            self._velocity_fbo.back,
            self._pressure_fbo,
            self._simulation_obstacle_fbo,
            self._cell,
            has_obstacles=self._has_obstacles
        )
        self._velocity_fbo.end()

    def _advect_density(self) -> None:
        """Advect & dissipate density field."""
        advect_step = self._dt * (self.settings.speed + self.settings.density.speed_offset)
        dissipation = self._calculate_dissipation(self._dt, self.settings.density.fade_time)

        self._density_fbo.swap()
        self._density_fbo.begin()
        self._advect_shader.use(
            self._density_fbo.back,    # Source density
            self._velocity_fbo,        # Velocity
            self._density_obstacle_fbo, # Obstacles (full density resolution)
            self._field_aspect,
            advect_step,
            dissipation,
            has_obstacles=self._has_obstacles
        )
        self._density_fbo.end()

    def _dampen(self, fbo: SwapFbo, threshold: float, dampen_time: float,
               delta_time: float, include_alpha: bool) -> None:
        """Exponential drag on magnitude excess above threshold.

        Args:
            fbo: Field to dampen (ping-pong swap handled by FlowUtil)
            threshold: Magnitude below which values are untouched
            dampen_time: Seconds for excess to decay to ~1%. 0=off
            delta_time: Frame delta time for frame-rate independence
            include_alpha: True for density (RGBA magnitude), False for velocity (RGB)
        """
        if dampen_time <= 0.0:
            return
        factor = pow(0.01, delta_time / dampen_time)
        FlowUtil.dampen(fbo, threshold, factor, include_alpha)

    # ========== Deferred ==========

    def _apply_wall_modes(self) -> None:
        """Apply the wall settings to all field FBOs (GL thread only)."""
        self._density_fbo.set_wrap(*_WALL_GL[self.settings.density.wall])
        self._temperature_fbo.set_wrap(*_WALL_GL[self.settings.temperature.wall])
        self._pressure_fbo.set_wrap(*_WALL_GL[self.settings.pressure.wall])

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
        s = self.settings
        self._dt = self._time_step.update(s.time_step, dt, s.fixed_fps, s.fps_smoothing)

    @staticmethod
    def _align32(v: int) -> int:
        """Round up to the nearest multiple of 32."""
        return (v + 31) & ~31

    @staticmethod
    def _calculate_dissipation(delta_time: float, decay_time: float) -> float:
        """Calculate frame-rate independent decay multiplier.

        Returns pow(0.01, dt / decay_time) -- field reaches 1% after decay_time seconds.
        """
        return pow(0.01, delta_time / max(0.001, decay_time))

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
        if not self._allocated or self.settings.temperature.buoyancy == 0.0:      # nothing reads it
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
        """R16F pressure field."""
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
        """RG16F buoyancy force field."""
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
                'buoyancy': Port(self._buoyancy_fbo, TextureKind.VELOCITY),
                'expansion': Port(self._expansion_fbo, TextureKind.SCALAR),
                'divergence': Port(self._divergence_fbo, TextureKind.SCALAR),
                'pressure': Port(self._pressure_fbo, TextureKind.SCALAR),
                'temperature': Port(self._temperature_fbo, TextureKind.SCALAR),
                'density': Port(self._density_fbo, TextureKind.IMAGE,
                                color_space=ColorSpace.OKLAB if self._oklab else ColorSpace.RGB)}

    @property
    def sim_width(self) -> int:
        """Current simulation resolution width (aligned to 16)."""
        return self._simulation_width

    @property
    def sim_height(self) -> int:
        """Current simulation resolution height (aligned to 16)."""
        return self._simulation_height
