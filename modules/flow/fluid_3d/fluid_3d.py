"""Fluid3D - 3D Navier-Stokes fluid simulation on the GPU.

Every field is a GL_TEXTURE_3D on one grid of cubic cells: the velocity (RGBA16F), the temperature
and the pressure (R16F) and the obstacle (R8) at the simulation's size, `layers` deep. The
density is its carrier's (`density.carrier`): the grid's (`GridDensity`, RGBA16F, its color baked in at
injection, on the same grid or `density.size` times finer in x and y, sampled by uv wherever a pass reads
it against the simulation's fields) or particles' (`ParticleDensity`), moved through the same velocity.

Pipeline:
    1. Dampen velocity (magnitude clamping)
    2. Advect velocity (3D self-advection; every advection semi-Lagrangian or MacCormack, `advection`)
    3. Apply viscosity (3D Jacobi diffusion solver)
    4. Confine vorticity (3D curl + confinement force)
    5. Advect temperature
    6. Apply buoyancy force
    7. Enforce incompressibility (divergence -> Jacobi solve -> gradient subtraction)
    8. Advect density
    9. Composite the density -> 2D output

The walls are per-field wrap modes (no explicit border obstacles):
    Velocity:          GL_CLAMP_TO_BORDER(0,0,0,0)  -> no-slip / Dirichlet
    Density, pressure, temperature: by their `wall`; the z walls by the volume's
    Obstacle:          GL_CLAMP_TO_BORDER(1,0,0,0)   -> OOB = obstacle

The inputs are the bridge's two systems: the boundary, the body in the water, whose velocity the
water is dragged toward where the body is (`drag_velocity`, never past it); and the body force, an
acceleration added × dt (`add_acceleration`). The densities and the temperatures are sources, added
× dt, each side at its own gain (`add_boundary_density`, `add_body_force_density`,
`add_boundary_temperature`, `add_body_force_temperature`). Every input goes into the gaussian birth
slab at `volume.injection_layer`.
"""
import math

import numpy as np
from OpenGL.GL import GL_CLAMP_TO_BORDER, GL_CLAMP_TO_EDGE, GL_NEAREST, GL_R16F, GL_R32F, GL_R8, GL_RGBA16F, GL_SHADER_IMAGE_ACCESS_BARRIER_BIT, GL_TEXTURE_FETCH_BARRIER_BIT, glMemoryBarrier

from modules.gl import SwapFbo, Texture, Texture3D, SwapTexture3D, GpuTimer
from ..wall_mode import WallMode
from .fluid_settings import Carrier, PressureSolver, Fluid3DSettings, VolumeSettings, DensitySettings, TemperatureSettings, PressureSettings, VelocitySettings, ParticleSettings
from .multigrid_pressure import MultigridPressure
from ..shaders import AddBoolean
from ..flow_util import FlowUtil
from ..port import Port, TextureKind
from ..time_step import REFERENCE_STEP, TimeStep, viscosity_iterations
from .density_carrier import CarrierStep, DensityCarrier
from .field_advection import FieldAdvection
from .grid_density import GridDensity
from .particle_density import ParticleDensity
from .shaders import (
    Divergence3D, Gradient3D,
    JacobiPressure3D, SorPressure3D, JacobiDiffusion3D,
    VorticityCurl3D, VorticityForce3D, Buoyancy3D,
    Inject3D, BoundaryVelocity3D, BodyForceAcceleration3D,
    Add3D,
    InjectBinary3D
)

# Combined memory barrier bits (cast to int for Pylance compatibility)
_BARRIER_FETCH_AND_IMAGE: int = int(GL_TEXTURE_FETCH_BARRIER_BIT) | int(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
_BARRIER_IMAGE: int = int(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

_WALL_GL: dict = {
    WallMode.OPEN:   (GL_CLAMP_TO_BORDER, (0.0, 0.0, 0.0, 0.0)),
    WallMode.CLOSED: (GL_CLAMP_TO_EDGE,   (0.0, 0.0, 0.0, 0.0)),
}

FLUID_PASSES: tuple[str, ...] = ('advect velocity', 'viscosity', 'vorticity', 'advect temperature', 'buoyancy',
                                 'pressure', 'density', 'composite')     # a step's passes, each a GPU timer
DRAG_FLOOR: float = 0.02    # widths/s: the body's speed below which it drags nothing, so a dead trail pulls nothing toward 0

import logging
logger = logging.getLogger(__name__)


# ── the drag's arithmetic, the shader's in numpy ─────────────────────────

def drag_share(delta_time: float, drag_time: float) -> float:
    """The share of the gap to the body's velocity the water closes in one step of `delta_time` (s):
    1 − exp(−dt / drag_time), `drag_time` in ms; 0 ms sets the velocity at once."""
    if drag_time <= 0.0:
        return 1.0
    return 1.0 - math.exp(-delta_time * 1000.0 / drag_time)


def drag_gate(speed: np.ndarray) -> np.ndarray:
    """Where the body is: its speed (widths/s) over the floor, at most 1."""
    return np.minimum(speed / DRAG_FLOOR, 1.0)


def drag_velocity_step(velocity: np.ndarray, boundary: np.ndarray, gain: float, share: float, weight: float = 1.0,
                       slab: float = 1.0) -> np.ndarray:
    """One drag step on (..., 3) velocities, in the volume's frame: v + (boundary × gain − v) × pull, with
    pull = share × gate × weight × slab, `boundary` the body's velocity. Mirrors boundary_velocity_3d.comp,
    without its w = −z."""
    gate: np.ndarray = drag_gate(np.linalg.norm(boundary, axis=-1, keepdims=True))
    pull: np.ndarray = share * gate * weight * slab
    return velocity + (boundary * gain - velocity) * pull


# ---------------------------------------------------------------------------
# 3D Fluid Simulation
# ---------------------------------------------------------------------------

class Fluid3D:
    """3D Navier-Stokes fluid simulation, every field a GL_TEXTURE_3D on one grid.

    Fields:
        velocity:    SwapTexture3D  RGBA16F  (u, v, w, spare)
        density:     the carrier's: GridDensity or ParticleDensity
        temperature: SwapTexture3D  R16F
        pressure:    SwapTexture3D  R32F
        obstacle:    Texture3D      R8

    Intermediate (single-buffer):
        divergence:       Texture3D R16F
        curl:             Texture3D RGBA16F  (wx, wy, wz, 0)
        buoyancy_force:   Texture3D RGBA16F  (Fx, Fy, Fz, 0)

    2D composited output:
        _output_texture:  Texture RGBA16F    (composited from the density)

    The fields leave through `ports`; `density` is the composited output.
    """

    def __init__(self, settings: Fluid3DSettings | None = None) -> None:
        self.settings: Fluid3DSettings = settings or Fluid3DSettings()

        # ---- Simulation dimensions and state ---
        self._simulation_width: int = 0
        self._simulation_height: int = 0
        self._output_width: int = 0       # the composited output, the layer's allocation
        self._output_height: int = 0
        self._depth: int = 0              # layers, every field's
        self._aspect: float = 1.0
        self._cell: float = 1.0           # the cubic cell's size in widths: 1 / the grid's width

        self._dt: float = REFERENCE_STEP
        self._time_step: TimeStep = TimeStep()

        # The slab's thickness in width units: cubic cells, so layers / width (advection: uRdx.z = 1/z_extent)
        self._z_extent: float = 1.0

        self._has_obstacles: bool = False

        self._allocated: bool = False

        self._reallocate_pending: bool = False
        self._carrier_pending: bool = False
        self._wall_dirty: bool = False

        self._gpu_timer: GpuTimer = GpuTimer("GPU fluid")
        self._pass_timers: dict[str, GpuTimer] = {name: GpuTimer(f"GPU fluid {name}") for name in FLUID_PASSES}

        # ---- Volumetric fields (SwapTexture3D for ping-pong) ----
        # Velocity always uses Dirichlet (v=0 at walls); others configurable via _apply_wall_modes().
        self._velocity: SwapTexture3D = SwapTexture3D(wrap=GL_CLAMP_TO_BORDER, border_color=(0.0, 0.0, 0.0, 0.0))
        self._temperature: SwapTexture3D = SwapTexture3D()
        self._temperature_clear: bool = False      # both buffers 0 since buoyancy went to 0
        self._pressure: SwapTexture3D = SwapTexture3D()
        # The MacCormack predictors, one per carried field, allocated at its first MacCormack step
        self._velocity_scratch: Texture3D = Texture3D(wrap=GL_CLAMP_TO_BORDER, border_color=(0.0, 0.0, 0.0, 0.0))
        self._temperature_scratch: Texture3D = Texture3D()
        # The density's carriers, the one `density.carrier` names allocated; the buoyancy's density while it gives none
        self._advection: FieldAdvection = FieldAdvection()
        self._grid_density: GridDensity = GridDensity(self.settings, self._advection)
        self._particle_density: ParticleDensity = ParticleDensity(self.settings)
        self._no_density: Texture3D = Texture3D()
        # Obstacle: CLAMP_TO_BORDER(1) = out-of-bounds = obstacle
        self._obstacle: Texture3D = Texture3D(interpolation=GL_NEAREST, wrap=GL_CLAMP_TO_BORDER, border_color=(1.0, 0.0, 0.0, 0.0))
        # 2D obstacle source (authoritative, never affected by a reallocation)
        self._obstacle_source: SwapFbo = SwapFbo(interpolation=GL_NEAREST, wrap=GL_CLAMP_TO_BORDER, border_color=(1.0, 0.0, 0.0, 0.0))

        # ---- Intermediate volumes (single buffer, no ping-pong) ----
        self._divergence_vol: Texture3D = Texture3D()
        # Curl is sampled at neighbor offsets by VorticityForce3D — border must return zero
        self._curl_vol: Texture3D = Texture3D(wrap=GL_CLAMP_TO_BORDER, border_color=(0.0, 0.0, 0.0, 0.0))
        self._buoyancy_force_vol: Texture3D = Texture3D()

        # ---- 2D composited output (for downstream render layers) ----
        self._output_texture: Texture = Texture()

        # ---- Compute shaders ----
        self._divergence_shader: Divergence3D = Divergence3D()
        self._gradient_shader: Gradient3D = Gradient3D()
        self._jacobi_pressure_shader: JacobiPressure3D = JacobiPressure3D()
        self._sor_pressure_shader: SorPressure3D = SorPressure3D()
        self._multigrid: MultigridPressure = MultigridPressure()
        self._jacobi_diffusion_shader: JacobiDiffusion3D = JacobiDiffusion3D()
        self._vorticity_curl_shader: VorticityCurl3D = VorticityCurl3D()
        self._vorticity_force_shader: VorticityForce3D = VorticityForce3D()
        self._buoyancy_shader: Buoyancy3D = Buoyancy3D()
        self._inject_shader: Inject3D = Inject3D()
        self._boundary_velocity_shader: BoundaryVelocity3D = BoundaryVelocity3D()
        self._body_force_acceleration_shader: BodyForceAcceleration3D = BodyForceAcceleration3D()
        self._add_shader: Add3D = Add3D()
        self._inject_binary_shader: InjectBinary3D = InjectBinary3D()
        self._add_boolean_shader: AddBoolean = AddBoolean()

        # Bind settings actions
        self.settings.bind(Fluid3DSettings.layers, lambda _: self._request_reallocate())
        reallocate_carrier = lambda _: setattr(self, '_carrier_pending', True)
        for field in (DensitySettings.size, DensitySettings.carrier):
            self.settings.density.bind(field, reallocate_carrier)
        for field in (ParticleSettings.count, ParticleSettings.slices):
            self.settings.particles.bind(field, reallocate_carrier)
        self.settings.bind(Fluid3DSettings.width, lambda _: self._request_reallocate())
        self.settings.bind(Fluid3DSettings.height, lambda _: self._request_reallocate())
        mark_wall = lambda _: setattr(self, '_wall_dirty', True)
        self.settings.density.bind(DensitySettings.wall, mark_wall)
        self.settings.temperature.bind(TemperatureSettings.wall, mark_wall)
        self.settings.pressure.bind(PressureSettings.wall, mark_wall)
        self.settings.volume.bind(VolumeSettings.wall, mark_wall)

    # ========== Allocation ==========

    def allocate(self, width: int, height: int) -> None:
        """Allocate all volumes and shaders.

        Args:
            width: The composited output's width, the layer's allocation
            height: The composited output's height
        """
        self._output_width = width
        self._output_height = height
        self._update_dimensions()

        self._advection.allocate()
        self._no_density.allocate(1, 1, 1, GL_RGBA16F)
        self._allocate_simulation_fields()
        self._allocate_carrier()

        # 2D obstacle source (output resolution, independent of the grid)
        self._obstacle_source.allocate(self._output_width, self._output_height, GL_R8)

        # 2D composited output
        self._output_texture.allocate(self._output_width, self._output_height, GL_RGBA16F)

        # Allocate shaders
        self._divergence_shader.allocate()
        self._gradient_shader.allocate()
        self._jacobi_pressure_shader.allocate()
        self._sor_pressure_shader.allocate()
        self._jacobi_diffusion_shader.allocate()
        self._vorticity_curl_shader.allocate()
        self._vorticity_force_shader.allocate()
        self._buoyancy_shader.allocate()
        self._inject_shader.allocate()
        self._boundary_velocity_shader.allocate()
        self._body_force_acceleration_shader.allocate()
        self._add_shader.allocate()
        self._inject_binary_shader.allocate()
        self._add_boolean_shader.allocate()

        self._allocated = True

    def _update_dimensions(self) -> None:
        """Recompute the grids' dimensions from the settings."""
        self._simulation_width = self._align32(self.settings.width)
        self._simulation_height = self._align32(self.settings.height)
        self._aspect = self._simulation_width / self._simulation_height if self._simulation_height > 0 else 1.0
        self._cell = 1.0 / max(1, self._simulation_width)
        self._depth = self.settings.layers
        self._z_extent = self._depth / max(1, self._simulation_width)

    def _spread(self) -> float:
        """The injection's gaussian sigma as a share of the slab's depth, from `injection_spread` in widths."""
        return max(self.settings.volume.injection_spread / max(self._z_extent, 1e-6), 0.001)

    @property
    def _carrier(self) -> DensityCarrier:
        """The density's carrier, by `density.carrier`."""
        return self._particle_density if self.settings.density.carrier == Carrier.PARTICLES else self._grid_density

    def _allocate_carrier(self) -> None:
        """(Re)allocate the carrier `density.carrier` names on the simulation's grid, the other one released."""
        for carrier in (self._grid_density, self._particle_density):
            carrier.deallocate()
        self._carrier.allocate(self._simulation_width, self._simulation_height, self._depth,
                               self._output_width, self._output_height)
        self._apply_wall_modes()

    def _allocate_simulation_fields(self) -> None:
        """(Re)allocate the simulation's volumes."""
        d = self._depth
        sim_w = self._simulation_width
        sim_h = self._simulation_height

        # Primary simulation fields
        self._velocity.allocate(sim_w, sim_h, d, GL_RGBA16F)
        self._temperature.allocate(sim_w, sim_h, d, GL_R16F)
        self._pressure.allocate(sim_w, sim_h, d, GL_R32F)     # a cell in widths makes it small, past half floats
        self._multigrid.allocate(sim_w, sim_h, d)
        self._obstacle.allocate(sim_w, sim_h, d, GL_R8)
        if self._has_obstacles:
            self._inject_binary_shader.use(self._obstacle_source, self._obstacle, mode=1)
            glMemoryBarrier(_BARRIER_IMAGE)

        # Intermediate volumes
        self._divergence_vol.allocate(sim_w, sim_h, d, GL_R16F)
        self._curl_vol.allocate(sim_w, sim_h, d, GL_RGBA16F)
        self._buoyancy_force_vol.allocate(sim_w, sim_h, d, GL_RGBA16F)

        self._apply_wall_modes()

    def deallocate(self) -> None:
        """Release all GPU resources."""
        self._gpu_timer.deallocate()
        for timer in self._pass_timers.values():
            timer.deallocate()
        self._velocity.deallocate()
        for carrier in (self._grid_density, self._particle_density):
            carrier.deallocate()
        self._no_density.deallocate()
        self._temperature.deallocate()
        self._pressure.deallocate()
        self._obstacle.deallocate()
        self._obstacle_source.deallocate()
        self._divergence_vol.deallocate()
        self._curl_vol.deallocate()
        self._buoyancy_force_vol.deallocate()
        self._output_texture.deallocate()
        for scratch in (self._velocity_scratch, self._temperature_scratch):
            scratch.deallocate()

        self._advection.deallocate()
        self._divergence_shader.deallocate()
        self._gradient_shader.deallocate()
        self._jacobi_pressure_shader.deallocate()
        self._sor_pressure_shader.deallocate()
        self._multigrid.deallocate()
        self._jacobi_diffusion_shader.deallocate()
        self._vorticity_curl_shader.deallocate()
        self._vorticity_force_shader.deallocate()
        self._buoyancy_shader.deallocate()
        self._inject_shader.deallocate()
        self._boundary_velocity_shader.deallocate()
        self._body_force_acceleration_shader.deallocate()
        self._add_shader.deallocate()
        self._inject_binary_shader.deallocate()
        self._add_boolean_shader.deallocate()

        self._allocated = False

    def _reload_shaders(self) -> None:
        """Hot-reload all shaders."""
        self._advection.reload()
        self._carrier.reload()
        self._divergence_shader.reload()
        self._gradient_shader.reload()
        self._jacobi_pressure_shader.reload()
        self._sor_pressure_shader.reload()
        self._multigrid.reload()
        self._jacobi_diffusion_shader.reload()
        self._vorticity_curl_shader.reload()
        self._vorticity_force_shader.reload()
        self._buoyancy_shader.reload()
        self._inject_shader.reload()
        self._boundary_velocity_shader.reload()
        self._body_force_acceleration_shader.reload()
        self._add_shader.reload()
        self._inject_binary_shader.reload()

    # ========== Update Pipeline ==========

    def reset(self) -> None:
        """Reset all simulation fields to zero."""
        if not self._allocated:
            return
        self._velocity.clear_all()
        self._carrier.reset()
        self._temperature.clear_all()
        self._temperature_clear = True
        self._pressure.clear_all()

    def update(self, dt: float) -> None:
        """Run one frame of the 3D fluid simulation pipeline; `dt` is the tick's measured length (s)."""
        if not self._allocated:
            return

        self._reload_shaders()
        self._handle_deferred_actions()

        # Per-frame state
        self._step_time(dt)

        self._gpu_timer.begin()

        # Simulation steps, each its own GPU timer inside the step's
        self._timed('advect velocity', self._advect_velocity)
        self._timed('viscosity', self._apply_viscosity)
        self._timed('vorticity', self._confine_vorticity)
        self._timed('advect temperature', self._advect_temperature)
        self._timed('buoyancy', self._apply_buoyancy)
        self._timed('pressure', self._enforce_incompressibility)
        self._timed('density', self._advect_density)
        self._timed('composite', self._composite_output)

        self._gpu_timer.end()

    def _timed(self, name: str, work) -> None:
        """`work` inside its pass's GPU timer, `GPU fluid <name>`."""
        timer: GpuTimer = self._pass_timers[name]
        timer.begin()
        work()
        timer.end()

    # ========== Pipeline Steps ==========

    def _add_force_to_velocity(self, force: Texture3D, strength: float = 1.0) -> None:
        """Add 3D force volume to velocity in-place (no swap needed)."""
        self._add_shader.use(self._velocity, force, strength)
        glMemoryBarrier(_BARRIER_IMAGE)

    def _advect_field(self, field: SwapTexture3D, scratch: Texture3D, velocity: Texture3D | None, step: float,
                      dissipation: float, internal_format: int, dampen: tuple[float, float] = (0.0, 1.0)) -> None:
        """Carry `field` one step along the velocity (`velocity`, or the field's own previous buffer for the
        velocity's self-advection), by `settings.advection`."""
        if self._advection.carry(self.settings.advection, field, scratch, velocity, self._obstacle, self._aspect,
                                 self._z_extent, step, dissipation, internal_format, self._has_obstacles, dampen=dampen):
            self._apply_wall_modes()

    def _advect_velocity(self) -> None:
        """Self-advect, dissipate and dampen the velocity field."""
        advect_step = self._dt * self.settings.velocity.self_advection
        dissipation = self._calculate_dissipation(self._dt, self.settings.velocity.fade_time)
        self._advect_field(self._velocity, self._velocity_scratch, None, advect_step, dissipation, GL_RGBA16F,
                           dampen=self._dampen_step())

    def _apply_viscosity(self) -> None:
        """Diffuse velocity via Jacobi viscosity solver."""
        if self.settings.velocity.viscosity <= 0.0:
            return

        viscosity_dt = self.settings.velocity.viscosity * self._dt       # widths² per second × s
        iterations = viscosity_iterations(self.settings.velocity.viscosity_iter, self._dt)

        result = self._jacobi_diffusion_shader.solve(
            self._velocity,
            self._velocity.back,
            self._obstacle,
            self._cell,
            viscosity_dt,
            total_iterations=iterations,
            iterations_per_dispatch=5,
            has_obstacles=self._has_obstacles
        )
        if result is not self._velocity:
            self._velocity.swap()
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

    def _confine_vorticity(self) -> None:
        """Vorticity confinement: the curl, then its force added to the velocity."""
        if self.settings.velocity.vorticity <= 0.0 or self.settings.velocity.vorticity_radius <= 0.0:
            return

        vorticity_radius = self.settings.velocity.vorticity_radius        # cells
        vorticity_force = self.settings.velocity.vorticity * self._cell * self._dt     # epsilon h (N x omega), Fedkiw 2001

        # a. Compute curl
        self._vorticity_curl_shader.use(
            self._velocity,
            self._obstacle,
            self._curl_vol,
            self._cell,
            vorticity_radius,
            has_obstacles=self._has_obstacles
        )
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

        # b. The confinement force, added to the velocity in the same pass
        self._vorticity_force_shader.use(
            self._curl_vol,
            self._obstacle,
            self._velocity,
            self._cell,
            vorticity_force,
            has_obstacles=self._has_obstacles,
        )
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

    def _advect_temperature(self) -> None:
        """Advect & dissipate the temperature scalar field."""
        if self.settings.temperature.buoyancy == 0.0:
            if not self._temperature_clear:     # cleared once; nothing adds to it while buoyancy is 0
                self._temperature.clear_all()
                self._temperature_clear = True
            return
        self._temperature_clear = False

        advect_step = self._dt * self.settings.speed
        dissipation = self._calculate_dissipation(self._dt, self.settings.temperature.fade_time)
        self._advect_field(self._temperature, self._temperature_scratch, self._velocity, advect_step, dissipation, GL_R16F)

    def _apply_buoyancy(self) -> None:
        """Apply buoyancy to velocity: F = gravity * (κρ − σ(T − T_ambient)).

        Dense density sinks along gravity; hot (fresh) density rises against it.
        The weight term acts wherever density is, so it stays active even
        with thermal buoyancy at 0.
        """
        temp = self.settings.temperature
        if temp.buoyancy == 0.0 and temp.sink == 0.0:
            return

        sigma = self._dt * temp.buoyancy
        kappa = self._dt * temp.sink
        gravity = (temp.gravity_x, temp.gravity_y, temp.gravity_z)

        weight: Texture3D | None = self._carrier.weight
        self._buoyancy_shader.use(
            self._temperature,
            weight if weight is not None else self._no_density,
            self._obstacle,
            self._buoyancy_force_vol,
            sigma, kappa, temp.ambient,
            gravity=gravity,
            has_obstacles=self._has_obstacles
        )
        glMemoryBarrier(_BARRIER_IMAGE)

        self._add_force_to_velocity(self._buoyancy_force_vol)

    def _enforce_incompressibility(self) -> None:
        """Pressure projection (divergence → Jacobi solve → gradient subtraction), the three passes on one set
        of walls: x and y closed or open by `pressure.wall`, z by `volume.wall`."""
        closed: tuple[bool, bool, bool] = (self.settings.pressure.wall == WallMode.CLOSED,
                                           self.settings.pressure.wall == WallMode.CLOSED,
                                           self.settings.volume.wall == WallMode.CLOSED)
        # a. Compute divergence
        self._divergence_shader.use(
            self._velocity,
            self._obstacle,
            self._divergence_vol,
            self._cell,
            has_obstacles=self._has_obstacles, closed=closed
        )
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

        # b. Solve Poisson equation for pressure, by the solver picked
        pressure: PressureSettings = self.settings.pressure
        iterations = pressure.iterations     # the grid sets what it needs, not the step
        if pressure.solver == PressureSolver.MULTIGRID:
            self._multigrid.solve(self._pressure, self._divergence_vol, self._obstacle, self._cell, pressure.cycles,
                                  has_obstacles=self._has_obstacles, closed=closed)
        elif pressure.solver == PressureSolver.SOR:
            self._sor_pressure_shader.solve(self._pressure, self._divergence_vol, self._obstacle, self._cell,
                                            iterations, pressure.relaxation,
                                            has_obstacles=self._has_obstacles, closed=closed)
        else:
            result = self._jacobi_pressure_shader.solve(
                self._pressure,
                self._pressure.back,
                self._divergence_vol,
                self._obstacle,
                self._cell,
                total_iterations=iterations,
                iterations_per_dispatch=5,
                has_obstacles=self._has_obstacles, closed=closed
            )
            if result is not self._pressure:
                self._pressure.swap()
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

        # c. Subtract pressure gradient from velocity
        self._velocity.swap()
        self._gradient_shader.use(
            self._velocity.back,
            self._pressure,
            self._obstacle,
            self._velocity,
            self._cell,
            has_obstacles=self._has_obstacles, closed=closed
        )
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

    def _advect_density(self) -> None:
        """Move the density one step along the velocity, faded and evaporated, by its carrier."""
        self._carrier.update(CarrierStep(
            velocity=self._velocity, obstacle=self._obstacle, has_obstacles=self._has_obstacles,
            aspect=self._aspect, z_extent=self._z_extent, dt=self._dt, step=self._dt * self.settings.speed,
            dissipation=self._calculate_dissipation(self._dt, self.settings.density.fade_time),
            evaporation=self.settings.density.evaporation * self._dt, advection=self.settings.advection))

    def _dampen_step(self) -> tuple[float, float]:
        """The velocity's dampen for this step, which its advection applies to the step's result: the threshold
        (widths/s) and the share of the excess kept, pow(0.01, dt / dampen_time); (0, 1) is none."""
        velocity: VelocitySettings = self.settings.velocity
        if velocity.dampen_time <= 0.0:
            return (0.0, 1.0)
        return (velocity.dampen_threshold, pow(0.01, self._dt / velocity.dampen_time))

    def _composite_output(self) -> None:
        """The density into the 2D output, by its carrier."""
        self._carrier.composite(self._output_texture)

    # ========== Deferred ==========

    def _apply_wall_modes(self) -> None:
        """Apply the wall settings to all field volumes (GL thread only)."""
        for temperature in (self._temperature, self._temperature_scratch):
            temperature.set_wrap(*_WALL_GL[self.settings.temperature.wall])
        self._pressure.set_wrap(*_WALL_GL[self.settings.pressure.wall])
        z_wall = _WALL_GL[self.settings.volume.wall]
        for vol in [self._velocity, self._temperature, self._pressure, self._velocity_scratch, self._temperature_scratch]:
            vol.set_wrap_z(*z_wall)
        self._carrier.set_walls(_WALL_GL[self.settings.density.wall], z_wall)

    def _handle_deferred_actions(self) -> None:
        """Process reset and reallocation requests queued from the UI thread."""
        if self._reallocate_pending:
            self._reallocate_pending = False
            old_sim = (self._simulation_width, self._simulation_height, self._depth)
            self._update_dimensions()
            if (self._simulation_width, self._simulation_height, self._depth) != old_sim:
                self._allocate_simulation_fields()
                self._carrier_pending = True

        if self._carrier_pending:
            self._carrier_pending = False
            self._allocate_carrier()

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

    def drag_velocity(self, texture: Texture, weight: float = 1.0) -> None:
        """The boundary: drag the velocity in the gaussian birth slab toward the body's velocity (x, y on
        the image plane, z toward the camera; widths/s) × boundary.gain, over boundary.drag_time, where
        the body moves (its speed above DRAG_FLOOR), never past it. The volume's w is -z: +W moves fluid
        away from the viewer (layer 0 is the front face). Without depth, z is 0, and so is w.
        """
        if not self._allocated:
            return
        boundary = self.settings.boundary
        self._boundary_velocity_shader.use(
            texture, self._velocity,
            self.settings.volume.injection_layer,
            self._spread(),
            boundary.gain, drag_share(self._dt, boundary.drag_time), weight, DRAG_FLOOR,
        )
        glMemoryBarrier(_BARRIER_IMAGE)

    def add_acceleration(self, texture: Texture, weight: float = 1.0) -> None:
        """The body force: add the acceleration (x, y on the image plane, z toward the camera; widths/s²)
        × body_force.gain × dt into the gaussian birth slab; w is -z, as the boundary's."""
        if not self._allocated:
            return
        self._body_force_acceleration_shader.use(
            texture, self._velocity,
            self.settings.volume.injection_layer,
            self._spread(),
            weight * self.settings.body_force.gain * self._dt,
        )
        glMemoryBarrier(_BARRIER_IMAGE)

    def add_boundary_density(self, texture: Texture, weight: float = 1.0) -> None:
        """The boundary's density (RGBA, widths/s times the color) × density.boundary_gain × dt into the slab,
        by the carrier."""
        self._add_density(texture, weight * self.settings.density.boundary_gain)

    def add_body_force_density(self, texture: Texture, weight: float = 1.0) -> None:
        """The body force's density (RGBA, widths/s² times the color) × density.body_force_gain × dt into the slab,
        by the carrier."""
        self._add_density(texture, weight * self.settings.density.body_force_gain)

    def _add_density(self, texture: Texture, gain: float) -> None:
        if not self._allocated:
            return
        volume = self.settings.volume
        self._carrier.add(texture, gain, self._dt, volume.injection_layer, self._spread())

    def add_boundary_temperature(self, texture: Texture, weight: float = 1.0) -> None:
        """The boundary's temperature (widths/s, signed by the color) × temperature.boundary_gain × dt into the slab."""
        if self.settings.temperature.buoyancy == 0.0:      # nothing reads it
            return
        self._add_source(texture, self._temperature, weight * self.settings.temperature.boundary_gain, GL_R16F)

    def add_body_force_temperature(self, texture: Texture, weight: float = 1.0) -> None:
        """The body force's temperature (widths/s², signed by the color) × temperature.body_force_gain × dt into the slab."""
        if self.settings.temperature.buoyancy == 0.0:      # nothing reads it
            return
        self._add_source(texture, self._temperature, weight * self.settings.temperature.body_force_gain, GL_R16F)

    def _add_source(self, texture: Texture, volume: SwapTexture3D, gain: float, internal_format: int) -> None:
        """A source: `texture` × gain × dt added into the volume's gaussian birth slab."""
        if not self._allocated:
            return
        self._inject_shader.use(
            texture, volume,
            self.settings.volume.injection_layer,
            self._spread(),
            gain * self._dt,
            internal_format=internal_format
        )
        glMemoryBarrier(_BARRIER_IMAGE)

    def set_obstacle(self, texture: Texture) -> None:
        """Replace obstacle volume with a 2D mask projected through all layers.

        Stores the 2D mask as the authoritative source, then projects it into the obstacle volume.

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
        self._inject_binary_shader.use(self._obstacle_source, self._obstacle, mode=1)
        glMemoryBarrier(_BARRIER_IMAGE)
        self._has_obstacles = True

    def add_obstacle(self, texture: Texture) -> None:
        """Add to obstacle volume (boolean OR with existing obstacles).

        Updates the 2D source and the obstacle volume.

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
        # Regenerate the volume from the updated source
        self._inject_binary_shader.use(self._obstacle_source, self._obstacle, mode=1)
        glMemoryBarrier(_BARRIER_IMAGE)
        self._has_obstacles = True

    def clear_obstacles(self) -> None:
        """Clear all obstacles."""
        if not self._allocated:
            return
        FlowUtil.zero(self._obstacle_source)
        self._obstacle.clear()
        glMemoryBarrier(_BARRIER_IMAGE)
        self._has_obstacles = False

    # ========== Outputs ==========

    @property
    def density(self) -> Texture:
        """2D composited density output for downstream render layers."""
        return self._output_texture

    @property
    def ports(self) -> dict[str, Port]:
        """Its fields in the order a step reaches them, each a volume in a box `_z_extent` deep, the 2D
        obstacle as it comes, and the carrier's (the grid's density; none for the particles)."""
        extent: float = self._z_extent

        def volume(texture: Texture3D, kind: TextureKind) -> Port:
            return Port(texture, kind, extent)

        return {'obstacle_source': Port(self._obstacle_source, TextureKind.SCALAR),
                'obstacle': volume(self._obstacle, TextureKind.SCALAR),
                'velocity': volume(self._velocity, TextureKind.VELOCITY),
                'vorticity': volume(self._curl_vol, TextureKind.VELOCITY),
                'temperature': volume(self._temperature, TextureKind.SCALAR),
                'buoyancy': volume(self._buoyancy_force_vol, TextureKind.VELOCITY),
                'divergence': volume(self._divergence_vol, TextureKind.SCALAR),
                'pressure': volume(self._pressure, TextureKind.SCALAR),
                **self._carrier.ports}
