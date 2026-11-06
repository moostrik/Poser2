"""Fluid3D - 3D Navier-Stokes fluid simulation on the GPU.

Density is stored as GL_TEXTURE_2D_ARRAY at output resolution for optimal
per-layer 2D tiling (RGBA16F, colour baked in at injection). Velocity,
pressure, temperature, and obstacles are GL_TEXTURE_3D at simulation scale.

Pipeline:
    1. Dampen velocity (magnitude clamping)
    2. Advect velocity (3D self-advection)
    3. Apply viscosity (3D Jacobi diffusion solver)
    4. Confine vorticity (3D curl + confinement force)
    5. Advect temperature
    6. Apply buoyancy force
    7. Enforce incompressibility (divergence -> Jacobi solve -> gradient subtraction)
    8. Advect density (GL_TEXTURE_2D_ARRAY, manual Z-lerp)
    9. Composite 3D density -> 2D output

The walls are per-field wrap modes (no explicit border obstacles):
    Velocity/density:  GL_CLAMP_TO_BORDER(0,0,0,0)  -> no-slip / Dirichlet
    Pressure/temp:     GL_CLAMP_TO_EDGE              -> zero-gradient / Neumann
    Obstacle:          GL_CLAMP_TO_BORDER(1,0,0,0)   -> OOB = obstacle
"""
import math
import time

from OpenGL.GL import GL_CLAMP_TO_BORDER, GL_CLAMP_TO_EDGE, GL_NEAREST, GL_R16F, GL_R8, GL_RGBA16F, GL_SHADER_IMAGE_ACCESS_BARRIER_BIT, GL_TEXTURE_FETCH_BARRIER_BIT, glMemoryBarrier

from modules.gl import SwapFbo, Texture, Texture3D, SwapTexture3D, SwapTexture2DArray, GpuTimer
from ..wall_mode import WallMode
from .fluid_settings import Fluid3DSettings, VolumeSettings, VelocitySettings, DensitySettings, TemperatureSettings, PressureSettings
from ..shaders import AddBoolean
from ..flow_util import FlowUtil
from ..port import OnDemand, Port, TextureKind
from ..time_step import REFERENCE_STEP, TimeStep, viscosity_iterations
from .shaders import (
    Advect3D, Advect3DDensity, Divergence3D, Gradient3D,
    JacobiPressure3D, JacobiDiffusion3D,
    VorticityCurl3D, VorticityForce3D, Buoyancy3DDensity,
    Inject3D, Inject3DDensity, InjectFlow3D,
    Dampen3D, Composite3DDensity, Add3D,
    InjectBinary3D, ReduceDensity
)

# Combined memory barrier bits (cast to int for Pylance compatibility)
_BARRIER_FETCH_AND_IMAGE: int = int(GL_TEXTURE_FETCH_BARRIER_BIT) | int(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
_BARRIER_IMAGE: int = int(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

_WALL_GL: dict = {
    WallMode.OPEN:   (GL_CLAMP_TO_BORDER, (0.0, 0.0, 0.0, 0.0)),
    WallMode.CLOSED: (GL_CLAMP_TO_EDGE,   (0.0, 0.0, 0.0, 0.0)),
}

import logging
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 3D Fluid Simulation
# ---------------------------------------------------------------------------

class Fluid3D:
    """3D Navier-Stokes fluid simulation with GL_TEXTURE_2D_ARRAY density.

    Density is stored as SwapTexture2DArray (GL_TEXTURE_2D_ARRAY) for
    optimal per-layer 2D tiling at high resolutions. Density advection,
    injection, buoyancy, and compositing use dedicated shaders with
    sampler2DArray / image2DArray bindings.

    Fields:
        velocity:    SwapTexture3D      RGBA16F  (u, v, w, spare)
        density:     SwapTexture2DArray RGBA16F  (r, g, b, a)
        temperature: SwapTexture3D      R16F
        pressure:    SwapTexture3D      R16F
        obstacle:    Texture3D          R8

    Intermediate (single-buffer):
        divergence:       Texture3D R16F
        curl:             Texture3D RGBA16F  (wx, wy, wz, 0)
        vorticity_force:  Texture3D RGBA16F  (Fx, Fy, Fz, 0)
        buoyancy_force:   Texture3D RGBA16F  (Fx, Fy, Fz, 0)

    2D composited output:
        _output_texture:  Texture RGBA16F    (composited from 3D density)

    The fields leave through `ports`; `density` is the composited output.
    """

    def __init__(self, settings: Fluid3DSettings | None = None) -> None:
        self.settings: Fluid3DSettings = settings or Fluid3DSettings()

        # ---- Simulation dimensions and state ---
        self._simulation_width: int = 0
        self._simulation_height: int = 0
        self._density_width: int = 0
        self._density_height: int = 0
        self._depth: int = 0          # density layers
        self._sim_depth: int = 0      # velocity/pressure/temperature layers
        self._aspect: float = 1.0

        self._dt: float = REFERENCE_STEP
        self._time_step: TimeStep = TimeStep()

        # Z metric, recomputed per update from depth.spacing (one physical slab, two layer counts)
        self._z_spacing: float = 1.0  # sim-grid dz in sim texels (solvers)
        self._z_extent: float = 1.0   # slab thickness in width units (advection: uRdx.z = 1/z_extent)

        self._has_obstacles: bool = False

        # Tracked-head eye offset for the composite (set_eye, polled per frame)
        self._eye_person: tuple[float, float] = (0.0, 0.0)

        self._allocated: bool = False

        self._reallocate_pending: bool = False
        self._wall_dirty: bool = False

        self._gpu_timer: GpuTimer = GpuTimer("GPU fluid")

        # ---- Volumetric fields (SwapTexture3D for ping-pong) ----
        # Velocity always uses Dirichlet (v=0 at walls); others configurable via _apply_wall_modes().
        self._velocity: SwapTexture3D = SwapTexture3D(wrap=GL_CLAMP_TO_BORDER, border_color=(0.0, 0.0, 0.0, 0.0))
        # Density uses GL_TEXTURE_2D_ARRAY for optimal per-layer 2D tiling.
        self._density: SwapTexture2DArray = SwapTexture2DArray()
        self._temperature: SwapTexture3D = SwapTexture3D()
        self._pressure: SwapTexture3D = SwapTexture3D()
        # Obstacle: CLAMP_TO_BORDER(1) = out-of-bounds = obstacle
        self._simulation_obstacle: Texture3D = Texture3D(interpolation=GL_NEAREST, wrap=GL_CLAMP_TO_BORDER, border_color=(1.0, 0.0, 0.0, 0.0))
        # Density-resolution obstacle (for density advection when grid_scale < 1)
        self._density_obstacle: Texture3D = Texture3D(interpolation=GL_NEAREST, wrap=GL_CLAMP_TO_BORDER, border_color=(1.0, 0.0, 0.0, 0.0))
        # 2D obstacle source (authoritative, never affected by depth/sim_scale changes)
        self._obstacle_source: SwapFbo = SwapFbo(interpolation=GL_NEAREST, wrap=GL_CLAMP_TO_BORDER, border_color=(1.0, 0.0, 0.0, 0.0))

        # ---- Intermediate volumes (single buffer, no ping-pong) ----
        self._divergence_vol: Texture3D = Texture3D()
        # Curl is sampled at neighbor offsets by VorticityForce3D — border must return zero
        self._curl_vol: Texture3D = Texture3D(wrap=GL_CLAMP_TO_BORDER, border_color=(0.0, 0.0, 0.0, 0.0))
        self._vorticity_force_vol: Texture3D = Texture3D()
        self._buoyancy_force_vol: Texture3D = Texture3D()

        # ---- 2D composited output (for downstream render layers) ----
        self._output_texture: Texture = Texture()

        # ---- Compute shaders ----
        self._advect_shader: Advect3D = Advect3D()
        self._divergence_shader: Divergence3D = Divergence3D()
        self._gradient_shader: Gradient3D = Gradient3D()
        self._jacobi_pressure_shader: JacobiPressure3D = JacobiPressure3D()
        self._jacobi_diffusion_shader: JacobiDiffusion3D = JacobiDiffusion3D()
        self._vorticity_curl_shader: VorticityCurl3D = VorticityCurl3D()
        self._vorticity_force_shader: VorticityForce3D = VorticityForce3D()
        self._inject_shader: Inject3D = Inject3D()
        self._inject_flow_shader: InjectFlow3D = InjectFlow3D()
        self._dampen_shader: Dampen3D = Dampen3D()
        # Density-specific shaders (GL_TEXTURE_2D_ARRAY bindings)
        self._advect_density_shader: Advect3DDensity = Advect3DDensity()
        self._buoyancy_density_shader: Buoyancy3DDensity = Buoyancy3DDensity()
        self._inject_density_shader: Inject3DDensity = Inject3DDensity()
        self._composite_density_shader: Composite3DDensity = Composite3DDensity()
        # The density as the monitors read it: at the simulation's width and height, a 3D texture, refreshed
        # once per step and only when a view asks (`_density_view_refresh`, the port's `prepare`)
        self._density_view: Texture3D = Texture3D()
        self._reduce_density_shader: ReduceDensity = ReduceDensity()
        self._density_view_refresh: OnDemand = OnDemand(self._reduce_density)
        self._add_shader: Add3D = Add3D()
        self._inject_binary_shader: InjectBinary3D = InjectBinary3D()
        self._add_boolean_shader: AddBoolean = AddBoolean()

        # Bind settings actions
        self.settings.volume.bind(VolumeSettings.layers, lambda _: self._request_reallocate())
        self.settings.volume.bind(VolumeSettings.sim_layers, lambda _: self._request_reallocate())
        self.settings.bind(Fluid3DSettings.width, lambda _: self._request_reallocate())
        self.settings.bind(Fluid3DSettings.height, lambda _: self._request_reallocate())
        mark_wall = lambda _: setattr(self, '_wall_dirty', True)
        self.settings.density.bind(DensitySettings.wall, mark_wall)
        self.settings.temperature.bind(TemperatureSettings.wall, mark_wall)
        self.settings.pressure.bind(PressureSettings.wall, mark_wall)
        self.settings.volume.bind(VolumeSettings.wall, mark_wall)

    # ========== Allocation ==========

    def allocate(self, width: int, height: int) -> None:
        """Allocate all 3D volumes and shaders.

        Args:
            width: Full output resolution width (XY)
            height: Full output resolution height (XY)
        """
        self._density_width = width
        self._density_height = height
        self._update_simulation_dimensions()

        self._allocate_simulation_fields()
        self._allocate_density_fields()

        # 2D obstacle source (full output resolution, depth-independent)
        self._obstacle_source.allocate(self._density_width, self._density_height, GL_R8)

        # 2D composited output
        self._output_texture.allocate(self._density_width, self._density_height, GL_RGBA16F)

        # Allocate shaders
        self._advect_shader.allocate()
        self._divergence_shader.allocate()
        self._gradient_shader.allocate()
        self._jacobi_pressure_shader.allocate()
        self._jacobi_diffusion_shader.allocate()
        self._vorticity_curl_shader.allocate()
        self._vorticity_force_shader.allocate()
        self._inject_shader.allocate()
        self._inject_flow_shader.allocate()
        self._dampen_shader.allocate()
        self._advect_density_shader.allocate()
        self._buoyancy_density_shader.allocate()
        self._inject_density_shader.allocate()
        self._composite_density_shader.allocate()
        self._reduce_density_shader.allocate()
        self._add_shader.allocate()
        self._inject_binary_shader.allocate()
        self._add_boolean_shader.allocate()

        self._allocated = True

    def _update_simulation_dimensions(self) -> None:
        """Recompute simulation dimensions from current settings."""
        self._simulation_width = self._align32(self.settings.width)
        self._simulation_height = self._align32(self.settings.height)
        self._aspect = self._simulation_width / self._simulation_height if self._simulation_height > 0 else 1.0
        self._grid_scale = self._simulation_width / self._density_width if self._density_width > 0 else 1.0
        self._depth = self.settings.volume.layers
        self._sim_depth = self.settings.volume.sim_layers

    def _allocate_density_fields(self) -> None:
        """(Re)allocate density-resolution volumes (full output resolution × depth)."""
        self._density.allocate(self._density_width, self._density_height, self._depth, GL_RGBA16F)
        self._density_obstacle.allocate(self._density_width, self._density_height, self._depth, GL_R8)
        self._regenerate_obstacle_volumes()
        self._allocate_density_view()

    def _allocate_simulation_fields(self) -> None:
        """(Re)allocate simulation-resolution 3D volumes."""
        d = self._sim_depth
        sim_w = self._simulation_width
        sim_h = self._simulation_height

        # Primary simulation fields
        self._velocity.allocate(sim_w, sim_h, d, GL_RGBA16F)
        self._temperature.allocate(sim_w, sim_h, d, GL_R16F)
        self._pressure.allocate(sim_w, sim_h, d, GL_R16F)
        self._simulation_obstacle.allocate(sim_w, sim_h, d, GL_R8)
        self._regenerate_obstacle_volumes()

        # Intermediate volumes
        self._divergence_vol.allocate(sim_w, sim_h, d, GL_R16F)
        self._curl_vol.allocate(sim_w, sim_h, d, GL_RGBA16F)
        self._vorticity_force_vol.allocate(sim_w, sim_h, d, GL_RGBA16F)
        self._buoyancy_force_vol.allocate(sim_w, sim_h, d, GL_RGBA16F)

        self._apply_wall_modes()
        self._allocate_density_view()

    def _allocate_density_view(self) -> None:
        """(Re)allocate the density as the monitors read it: the simulation's width and height, the
        density's layers. Both allocations call it, since it follows both."""
        if self._simulation_width <= 0 or self._depth <= 0:
            return
        size: tuple[int, int, int] = (self._simulation_width, self._simulation_height, self._depth)
        view: Texture3D = self._density_view
        if view.allocated and (view.width, view.height, view.depth) == size:
            return
        view.deallocate()
        view.allocate(*size, GL_RGBA16F)
        self._density_view_refresh.mark()

    def _reduce_density(self) -> None:
        """The density averaged down into `_density_view`; the port's `prepare`, run once per step at most."""
        self._reduce_density_shader.use(self._density, self._density_view)
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

    def _regenerate_obstacle_volumes(self) -> None:
        """Project 2D obstacle source into both 3D obstacle volumes."""
        if not self._has_obstacles:
            return
        self._inject_binary_shader.use(self._obstacle_source, self._simulation_obstacle, mode=1)
        self._inject_binary_shader.use(self._obstacle_source, self._density_obstacle, mode=1)
        glMemoryBarrier(_BARRIER_IMAGE)

    def deallocate(self) -> None:
        """Release all GPU resources."""
        self._gpu_timer.deallocate()
        self._velocity.deallocate()
        self._density.deallocate()
        self._temperature.deallocate()
        self._pressure.deallocate()
        self._density_obstacle.deallocate()
        self._simulation_obstacle.deallocate()
        self._obstacle_source.deallocate()
        self._divergence_vol.deallocate()
        self._curl_vol.deallocate()
        self._vorticity_force_vol.deallocate()
        self._buoyancy_force_vol.deallocate()
        self._output_texture.deallocate()

        self._advect_shader.deallocate()
        self._divergence_shader.deallocate()
        self._gradient_shader.deallocate()
        self._jacobi_pressure_shader.deallocate()
        self._jacobi_diffusion_shader.deallocate()
        self._vorticity_curl_shader.deallocate()
        self._vorticity_force_shader.deallocate()
        self._inject_shader.deallocate()
        self._inject_flow_shader.deallocate()
        self._dampen_shader.deallocate()
        self._advect_density_shader.deallocate()
        self._buoyancy_density_shader.deallocate()
        self._inject_density_shader.deallocate()
        self._composite_density_shader.deallocate()
        self._reduce_density_shader.deallocate()
        self._density_view.deallocate()
        self._add_shader.deallocate()
        self._inject_binary_shader.deallocate()
        self._add_boolean_shader.deallocate()

        self._allocated = False

    def _reload_shaders(self) -> None:
        """Hot-reload all shaders."""
        self._advect_shader.reload()
        self._divergence_shader.reload()
        self._gradient_shader.reload()
        self._jacobi_pressure_shader.reload()
        self._jacobi_diffusion_shader.reload()
        self._vorticity_curl_shader.reload()
        self._vorticity_force_shader.reload()
        self._inject_shader.reload()
        self._inject_flow_shader.reload()
        self._dampen_shader.reload()
        self._advect_density_shader.reload()
        self._buoyancy_density_shader.reload()
        self._inject_density_shader.reload()
        self._composite_density_shader.reload()
        self._reduce_density_shader.reload()
        self._add_shader.reload()
        self._inject_binary_shader.reload()

    # ========== Update Pipeline ==========

    def reset(self) -> None:
        """Reset all simulation fields to zero."""
        if not self._allocated:
            return
        self._velocity.clear_all()
        self._density.clear_all()
        self._temperature.clear_all()
        self._pressure.clear_all()

    def update(self, dt: float) -> None:
        """Run one frame of the 3D fluid simulation pipeline; `dt` is the tick's measured length (s)."""
        if not self._allocated:
            return

        self._reload_shaders()
        self._handle_deferred_actions()

        # Per-frame state
        self._step_time(dt)

        # Z metric from live settings: solvers see sim-grid cell size, advection sees slab thickness
        spacing = max(1.0, self.settings.volume.spacing)
        self._z_spacing = spacing
        self._z_extent = (self._sim_depth * spacing) / max(1, self._simulation_width)

        self._gpu_timer.begin()

        # Dampen velocity (clean input for all steps)
        vel: VelocitySettings = self.settings.velocity
        self._dampen(self._velocity, vel.dampen_threshold, vel.dampen_time, self._dt, include_alpha=False)

        # Simulation steps
        self._advect_velocity()
        self._apply_viscosity()
        self._confine_vorticity()
        self._advect_temperature()
        self._apply_buoyancy()
        self._enforce_incompressibility()
        self._advect_density()
        # Density dampen disabled — not meaningful for 3D and wastes a full-volume pass
        self._composite_output()

        self._gpu_timer.end()
        self._density_view_refresh.mark()

    # ========== Pipeline Steps ==========

    def _add_force_to_velocity(self, force: Texture3D, strength: float = 1.0) -> None:
        """Add 3D force volume to velocity in-place (no swap needed)."""
        self._add_shader.use(self._velocity, force, strength)
        glMemoryBarrier(_BARRIER_IMAGE)

    def _advect_velocity(self) -> None:
        """Self-advect & dissipate velocity field."""
        advect_step = self._dt * self.settings.velocity.self_advection
        dissipation = self._calculate_dissipation(self._dt, self.settings.velocity.fade_time)

        self._velocity.swap()
        self._advect_shader.advect(
            self._velocity.back,
            self._velocity,
            self._velocity.back,
            self._simulation_obstacle,
            self._aspect, self._z_extent,
            advect_step, dissipation,
            internal_format=GL_RGBA16F,
            has_obstacles=self._has_obstacles
        )
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

    def _apply_viscosity(self) -> None:
        """Diffuse velocity via Jacobi viscosity solver."""
        if self.settings.velocity.viscosity <= 0.0:
            return

        viscosity_dt = self.settings.velocity.viscosity * (self._grid_scale ** 2) * self._dt
        iterations = viscosity_iterations(self.settings.velocity.viscosity_iter, self._dt)

        result = self._jacobi_diffusion_shader.solve(
            self._velocity,
            self._velocity.back,
            self._simulation_obstacle,
            self._grid_scale, self._aspect, self._z_spacing,
            viscosity_dt,
            total_iterations=iterations,
            iterations_per_dispatch=5,
            has_obstacles=self._has_obstacles
        )
        if result is not self._velocity:
            self._velocity.swap()
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

    def _confine_vorticity(self) -> None:
        """Vorticity confinement (curl → force → add to velocity)."""
        if self.settings.velocity.vorticity <= 0.0 or self.settings.velocity.vorticity_radius <= 0.0:
            return

        vorticity_radius = self.settings.velocity.vorticity_radius * self._grid_scale
        vorticity_force = self.settings.velocity.vorticity * self._dt

        # a. Compute curl
        self._vorticity_curl_shader.use(
            self._velocity,
            self._simulation_obstacle,
            self._curl_vol,
            self._grid_scale, self._aspect, self._z_spacing,
            vorticity_radius,
            has_obstacles=self._has_obstacles
        )
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

        # b. Compute confinement force
        self._vorticity_force_shader.use(
            self._curl_vol,
            self._simulation_obstacle,
            self._vorticity_force_vol,
            self._grid_scale, self._aspect, self._z_spacing,
            vorticity_force,
            has_obstacles=self._has_obstacles,
        )
        glMemoryBarrier(_BARRIER_IMAGE)

        # c. Add force to velocity
        self._add_force_to_velocity(self._vorticity_force_vol)

    def _advect_temperature(self) -> None:
        """Advect & dissipate the temperature scalar field."""
        if self.settings.temperature.buoyancy == 0.0:
            self._temperature.clear_all()
            return

        advect_step = self._dt * self.settings.speed
        dissipation = self._calculate_dissipation(self._dt, self.settings.temperature.fade_time)

        self._temperature.swap()
        self._advect_shader.advect(
            self._temperature.back,
            self._temperature,
            self._velocity,
            self._simulation_obstacle,
            self._aspect, self._z_extent,
            advect_step, dissipation,
            internal_format=GL_R16F,
            has_obstacles=self._has_obstacles
        )
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

    def _apply_buoyancy(self) -> None:
        """Apply buoyancy to velocity: F = gravity * (κρ − σ(T − T_ambient)).

        Dense density sinks along gravity; hot (fresh) density rises against it.
        The weight term acts wherever density is, so it stays active even
        with thermal buoyancy at 0.
        """
        temp = self.settings.temperature
        if temp.buoyancy == 0.0 and temp.sink == 0.0:
            return

        sigma = self._dt * self._grid_scale * temp.buoyancy
        kappa = self._dt * self._grid_scale * temp.sink
        gravity = (temp.gravity_x, temp.gravity_y, temp.gravity_z)

        self._buoyancy_density_shader.use(
            self._temperature,
            self._density,
            self._simulation_obstacle,
            self._buoyancy_force_vol,
            sigma, kappa, temp.ambient,
            gravity=gravity,
            has_obstacles=self._has_obstacles
        )
        glMemoryBarrier(_BARRIER_IMAGE)

        self._add_force_to_velocity(self._buoyancy_force_vol)

    def _enforce_incompressibility(self) -> None:
        """Pressure projection (divergence → Jacobi solve → gradient subtraction)."""
        # a. Compute divergence
        self._divergence_shader.use(
            self._velocity,
            self._simulation_obstacle,
            self._divergence_vol,
            self._grid_scale, self._aspect, self._z_spacing,
            has_obstacles=self._has_obstacles
        )
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

        # b. Solve Poisson equation for pressure
        iterations = self.settings.pressure.iterations     # the grid sets what it needs, not the step

        result = self._jacobi_pressure_shader.solve(
            self._pressure,
            self._pressure.back,
            self._divergence_vol,
            self._simulation_obstacle,
            self._grid_scale, self._aspect, self._z_spacing,
            total_iterations=iterations,
            iterations_per_dispatch=5,
            has_obstacles=self._has_obstacles
        )
        if result is not self._pressure:
            self._pressure.swap()
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

        # c. Subtract pressure gradient from velocity
        self._velocity.swap()
        self._gradient_shader.use(
            self._velocity.back,
            self._pressure,
            self._simulation_obstacle,
            self._velocity,
            self._grid_scale, self._aspect, self._z_spacing,
            has_obstacles=self._has_obstacles
        )
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

    def _advect_density(self) -> None:
        """Advect & dissipate density field."""
        advect_step = self._dt * self.settings.speed
        dissipation = self._calculate_dissipation(self._dt, self.settings.density.fade_time)

        self._density.swap()
        self._advect_density_shader.advect(
            self._density.back,
            self._density,
            self._velocity,
            self._density_obstacle,
            self._aspect, self._z_extent,
            advect_step, dissipation,
            internal_format=GL_RGBA16F,
            has_obstacles=self._has_obstacles,
            closed_z=self.settings.volume.wall == WallMode.CLOSED,
            evaporation=self.settings.density.evaporation * self._dt
        )
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

    def _dampen(self, volume: SwapTexture3D, threshold: float, dampen_time: float,
               delta_time: float, include_alpha: bool,
               internal_format: int = GL_RGBA16F) -> None:
        """Exponential drag on magnitude excess above threshold.

        Args:
            volume: Volumetric field to dampen (in-place via compute shader)
            threshold: Magnitude below which values are untouched
            dampen_time: Seconds for excess to decay to ~1%. 0=off
            delta_time: Frame delta time for frame-rate independence
            include_alpha: True for density (RGBA magnitude), False for velocity (RGB)
            internal_format: Image format for volume binding
        """
        if dampen_time <= 0.0:
            return
        factor = pow(0.01, delta_time / dampen_time)
        self._dampen_shader.use(
            volume, threshold, factor,
            include_alpha=include_alpha,
            internal_format=internal_format
        )
        glMemoryBarrier(_BARRIER_IMAGE)

    def set_eye(self, x: float, y: float) -> None:
        """Store the tracked-head eye offset (±1 range), polled by the composite."""
        self._eye_person = (x, y)

    def _composite_output(self) -> None:
        """Composite 3D density → 2D output with depth cues (virtual eye)."""
        d = self.settings.volume
        if d.drift_period > 0.0:
            # Lissajous orbit — never visibly repeats, period in seconds
            t = (2.0 * math.pi) * time.monotonic() / d.drift_period
            drift = (d.parallax * math.sin(t), 0.6 * d.parallax * math.cos(t * 0.77))
        else:
            drift = (d.parallax, 0.0)
        # Crossfade autonomous drift <-> head-coupled parallax
        k = d.eye_tracking
        eye = (drift[0] * (1.0 - k) + self._eye_person[0] * d.parallax * k,
               drift[1] * (1.0 - k) + self._eye_person[1] * d.parallax * k)
        self._composite_density_shader.use(
            self._density,
            self._output_texture,
            emission=d.emission,
            gamma=d.gamma,
            absorption=d.absorption,
            tint=d.tint,
            fog=d.fog,
            glow=d.glow,
            eye=eye,
            perspective=d.perspective,
            depth_blur=d.layer_blur,
            debug_depth=d.debug_layers,
        )
        glMemoryBarrier(_BARRIER_IMAGE)

    # ========== Deferred ==========

    def _apply_wall_modes(self) -> None:
        """Apply the wall settings to all field volumes (GL thread only)."""
        self._density.set_wrap(*_WALL_GL[self.settings.density.wall])
        self._temperature.set_wrap(*_WALL_GL[self.settings.temperature.wall])
        self._pressure.set_wrap(*_WALL_GL[self.settings.pressure.wall])
        z_wrap, z_color = _WALL_GL[self.settings.volume.wall]
        for vol in [self._velocity, self._temperature, self._pressure]:
            vol.set_wrap_z(z_wrap, z_color)

    def _handle_deferred_actions(self) -> None:
        """Process reset and reallocation requests queued from the UI thread."""
        if self._reallocate_pending:
            self._reallocate_pending = False
            old_w, old_h = self._simulation_width, self._simulation_height
            old_d, old_sd = self._depth, self._sim_depth
            self._update_simulation_dimensions()
            sim_changed = (self._simulation_width != old_w or self._simulation_height != old_h
                           or self._sim_depth != old_sd)
            depth_changed = self._depth != old_d
            if sim_changed:
                self._allocate_simulation_fields()
            if depth_changed:
                self._allocate_density_fields()

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

    # ========== Input Methods (2D -> 3D injection) ==========

    def add_velocity(self, texture: Texture, strength: float = 1.0) -> None:
        """Inject the 3D velocity (x, y on the image plane, z toward the camera) into the gaussian
        birth slab, gained by velocity.gain. The volume's w is -z: +W moves fluid away from the
        viewer (layer 0 is the front face). Without depth, z is 0, and so is w.
        """
        if not self._allocated:
            return
        self._inject_flow_shader.use(
            texture, self._velocity,
            self.settings.volume.injection_layer,
            self.settings.volume.injection_spread,
            strength * self.settings.velocity.gain * self._dt,
        )
        glMemoryBarrier(_BARRIER_IMAGE)

    def add_density(self, texture: Texture, strength: float = 1.0) -> None:
        """Inject 2D density texture into 3D density volume. Applies settings.density.gain and delta_time."""
        if not self._allocated:
            return
        self._inject_density_shader.use(
            texture, self._density,
            self.settings.volume.injection_layer,
            self.settings.volume.injection_spread,
            strength * self.settings.density.gain * self._dt,
            internal_format=GL_RGBA16F
        )
        glMemoryBarrier(_BARRIER_IMAGE)

    def add_temperature(self, texture: Texture, strength: float = 1.0) -> None:
        """Inject 2D temperature texture into 3D temperature volume. Applies settings.temperature.gain and delta_time."""
        if not self._allocated:
            return
        self._inject_shader.use(
            texture, self._temperature,
            self.settings.volume.injection_layer,
            self.settings.volume.injection_spread,
            strength * self.settings.temperature.gain * self._dt,
            internal_format=GL_R16F
        )
        glMemoryBarrier(_BARRIER_IMAGE)

    def set_obstacle(self, texture: Texture) -> None:
        """Replace obstacle volume with a 2D mask projected through all layers.

        Stores the 2D mask as the authoritative source, then projects into
        both sim-resolution and density-resolution 3D obstacle volumes.

        Args:
            texture: 2D obstacle mask (any channel > 0.5 = obstacle)
        """
        if not self._allocated:
            return
        # Replace 2D source (clear → OR with texture = texture)
        FlowUtil.zero(self._obstacle_source)
        self._obstacle_source.swap()
        self._obstacle_source.begin()
        self._add_boolean_shader.use(self._obstacle_source.back, texture)
        self._obstacle_source.end()
        # Project to 3D volumes
        self._inject_binary_shader.use(self._obstacle_source, self._simulation_obstacle, mode=1)
        self._inject_binary_shader.use(self._obstacle_source, self._density_obstacle, mode=1)
        glMemoryBarrier(_BARRIER_IMAGE)
        self._has_obstacles = True

    def add_obstacle(self, texture: Texture) -> None:
        """Add to obstacle volume (boolean OR with existing obstacles).

        Updates the 2D source and both 3D obstacle volumes.

        Args:
            texture: 2D obstacle mask to add (any channel > 0.5 = obstacle)
        """
        if not self._allocated:
            return
        # OR into 2D source
        self._obstacle_source.swap()
        self._obstacle_source.begin()
        self._add_boolean_shader.use(self._obstacle_source.back, texture)
        self._obstacle_source.end()
        # Regenerate 3D volumes from updated source
        self._inject_binary_shader.use(self._obstacle_source, self._simulation_obstacle, mode=1)
        self._inject_binary_shader.use(self._obstacle_source, self._density_obstacle, mode=1)
        glMemoryBarrier(_BARRIER_IMAGE)
        self._has_obstacles = True

    def clear_obstacles(self) -> None:
        """Clear all obstacles."""
        if not self._allocated:
            return
        FlowUtil.zero(self._obstacle_source)
        self._simulation_obstacle.clear()
        self._density_obstacle.clear()
        glMemoryBarrier(_BARRIER_IMAGE)
        self._has_obstacles = False

    # ========== Outputs ==========

    @property
    def density(self) -> Texture:
        """2D composited density output for downstream render layers."""
        return self._output_texture

    @property
    def ports(self) -> dict[str, Port]:
        """Its fields in the order a step reaches them, each a volume in a box `_z_extent` deep, and the
        2D obstacle as it comes. The density's port is its reduced view, refreshed when a view asks."""
        extent: float = self._z_extent

        def volume(texture: Texture3D, kind: TextureKind) -> Port:
            return Port(texture, kind, extent)

        return {'obstacle_source': Port(self._obstacle_source, TextureKind.SCALAR),
                'obstacle': volume(self._simulation_obstacle, TextureKind.SCALAR),
                'velocity': volume(self._velocity, TextureKind.VELOCITY),
                'vorticity': volume(self._curl_vol, TextureKind.VELOCITY),
                'vorticity_force': volume(self._vorticity_force_vol, TextureKind.VELOCITY),
                'temperature': volume(self._temperature, TextureKind.SCALAR),
                'buoyancy': volume(self._buoyancy_force_vol, TextureKind.VELOCITY),
                'divergence': volume(self._divergence_vol, TextureKind.SCALAR),
                'pressure': volume(self._pressure, TextureKind.SCALAR),
                'density': Port(self._density_view, TextureKind.IMAGE, extent, self._density_view_refresh)}
