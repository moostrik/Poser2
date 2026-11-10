"""Fluid3D - 3D Navier-Stokes fluid simulation on the GPU.

Every field is a GL_TEXTURE_3D on one grid of cubic cells: the velocity (RGBA16F), the temperature
and the pressure (R16F) and the obstacle (R8) at the simulation's size, `depth` deep. The
density is carried by the simulation density always (`SimulationDensity`, RGBA16F, its color baked in at
injection, on the same grid), and where enabled by the upscaled density (`UpscaledDensity`, its amount at
`upscaled`'s size, sampled by uv wherever a pass reads it against the simulation's fields, its color the
simulation density's) and the particles (`ParticleDensity`), all moved through the same velocity. The volume
drawn is the upscaled one where enabled, else the simulation's; the particles are laid over it.

Pipeline:
    1. Advect velocity (3D self-advection, on its result the guard and `max_speed`; every advection semi-Lagrangian
       or MacCormack, `advection`; each field's advection sets its non-finite values to 0 and counts them, `FiniteGuard`)
    2. (the top speed is the advection's)
    3. Apply viscosity (3D Jacobi diffusion solver)
    4. Confine vorticity (3D curl + confinement force)
    5. Advect temperature
    6. Apply buoyancy force
    7. Enforce incompressibility (divergence -> Jacobi solve -> gradient subtraction)
    8. Advect density
    9. Composite the density -> 2D output

The walls are per-field wrap modes (no explicit border obstacles):
    Velocity:          GL_CLAMP_TO_BORDER(0,0,0,0)  -> no-slip / Dirichlet
    Velocity z, density, pressure, temperature: by `wall`, every side alike
    Obstacle:          GL_CLAMP_TO_BORDER(1,0,0,0)   -> OOB = obstacle

The inputs are the bridge's two systems: the boundary, the body in the water, whose velocity the
water is dragged toward where the body is (`drag_velocity`, never past it); and the body force, an
acceleration added × dt (`add_acceleration`). The densities and the temperatures are sources, added
× dt, each side at its own gain (`add_boundary_density`, `add_body_force_density`,
`add_boundary_temperature`, `add_body_force_temperature`). Every input goes into the gaussian birth
slab at `input.layer`.
"""
import math
from dataclasses import replace

import numpy as np
from OpenGL.GL import GL_CLAMP_TO_BORDER, GL_CLAMP_TO_EDGE, GL_NEAREST, GL_R16F, GL_R32F, GL_R8, GL_RGBA16F, GL_SHADER_IMAGE_ACCESS_BARRIER_BIT, GL_TEXTURE_FETCH_BARRIER_BIT, glMemoryBarrier

from modules.gl import ComputeShader, SwapFbo, Texture, Texture3D, SwapTexture3D, GpuTimer
from ..wall_mode import WallMode
from .fluid_settings import Advection, Light, PressureSolver, Fluid3DSettings, PressureSettings, ParticleSettings, UpscaledDensitySettings
from .multigrid_pressure import MultigridPressure
from .face_velocity import FaceVelocity
from ..shaders import AddBoolean
from ..flow_util import FlowUtil
from ..port import LightPlace, Port, TextureKind
from ..time_step import REFERENCE_STEP, TimeStep, viscosity_iterations
from ..finite_guard import FiniteGuard, GuardSlot
from .density_carrier import CarrierStep, DensityCarrier
from .upscaled_density import UpscaledDensity
from .field_advection import FieldAdvection
from .simulation_density import SimulationDensity
from .particle_density import ParticleDensity
from .shaders import (AdvectFaces3D, FacesToCenters3D, ViscosityFaces3D, SEMI_LAGRANGIAN, PREDICTOR, CORRECTION,
                      
    Divergence3D, Gradient3D,
    JacobiPressure3D, SorPressure3D,
    VorticityCurl3D, VorticityForce3D, Buoyancy3D, AddToFaces3D,
    Inject3D,
    CompositeOver, Knee,
    CompositeLight, LightVolume3D, LightBlur3D, InjectMode, view_march,
    InjectBinary3D
)

WATER_CHANNELS: tuple[float, float, float] = (1.8, 1.0, 0.6)    # the water's relative extinction, red first; composite_3d's

# Combined memory barrier bits (cast to int for Pylance compatibility)
_BARRIER_FETCH_AND_IMAGE: int = int(GL_TEXTURE_FETCH_BARRIER_BIT) | int(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
_BARRIER_IMAGE: int = int(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

_WALL_GL: dict = {
    WallMode.OPEN:   (GL_CLAMP_TO_BORDER, (0.0, 0.0, 0.0, 0.0)),
    WallMode.CLOSED: (GL_CLAMP_TO_EDGE,   (0.0, 0.0, 0.0, 0.0)),
}

FLUID_PASSES: tuple[str, ...] = ('advect velocity', 'viscosity', 'vorticity', 'advect temperature', 'buoyancy', 'pressure',
                                 'faces to centers', 'density', 'light', 'composite')     # a step's passes, each a GPU timer
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


def soft_limit(velocity: np.ndarray, max_speed: float) -> np.ndarray:
    """The velocity's top speed, as the advection's `limit_speed`: (..., 3) widths/s untouched up to half of
    `max_speed`, above it bent smoothly toward it by a tanh, its direction kept; 0 no limit."""
    if max_speed <= 0.0:
        return velocity
    speed: np.ndarray = np.linalg.norm(velocity, axis=-1, keepdims=True)
    knee: float = 0.5 * max_speed
    span: float = max_speed - knee
    bent: np.ndarray = knee + span * np.tanh((speed - knee) / span)
    scale: np.ndarray = np.where(speed > knee, bent / np.maximum(speed, 1e-12), 1.0)
    return velocity * scale


def drag_velocity_step(velocity: np.ndarray, boundary: np.ndarray, gain: float, share: float, weight: float = 1.0,
                       slab: float = 1.0) -> np.ndarray:
    """One drag step on (..., 3) velocities, in the volume's frame: v + (boundary × gain − v) × pull, with
    pull = share × gate × weight × slab, `boundary` the body's velocity. Mirrors add_to_faces_3d.comp's drag,
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
        density:     SimulationDensity, with UpscaledDensity and ParticleDensity where enabled
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
        self._depth_scale: float = 1.0      # the body's z to the box's: its depth over the smaller of its width and height

        self._has_obstacles: bool = False

        self._allocated: bool = False

        self._reallocate_pending: bool = False
        self._densities_pending: bool = False
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
        self._temperature_scratch: Texture3D = Texture3D()
        # The density's carriers: the simulation density always, the upscaled and the particles by their `enabled`;
        # the particles drawn into their own image, then over the volume
        self._advection: FieldAdvection = FieldAdvection()
        # The fields' non-finite values, set to 0 where each is advected, counted and reported
        self._guard: FiniteGuard = FiniteGuard('fluid', ('velocity', 'density', 'temperature'))
        self._simulation_density: SimulationDensity = SimulationDensity(self.settings, self._advection)
        self._particle_density: ParticleDensity = ParticleDensity(self.settings)
        self._upscaled_density: UpscaledDensity = UpscaledDensity(self.settings, self._advection, self._simulation_density)
        self._particle_texture: Texture = Texture()
        self._over_shader: CompositeOver = CompositeOver()
        self._knee_shader: Knee = Knee()               # the output's highlights bent, not clipped
        # The light: how much reaches each cell of a grid at a quarter of the simulation's size, and its blur
        # for the scattering; allocated while `composite.shadow` is above 0
        self._light: Texture3D = Texture3D(wrap=GL_CLAMP_TO_EDGE)
        self._light_blur: tuple[Texture3D, Texture3D] = (Texture3D(wrap=GL_CLAMP_TO_EDGE), Texture3D(wrap=GL_CLAMP_TO_EDGE))
        self._light_shader: LightVolume3D = LightVolume3D()
        self._light_blur_shader: LightBlur3D = LightBlur3D()
        self._light_override: tuple[float, float, float] | None = None     # set_light's, over the settings
        self._light_shown: Texture3D = self._light      # the light the composite read last: direct, or scattered
        self._light_now: CompositeLight | None = None   # this step's light, `_update_light`'s, for the composite
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
        self._vorticity_force_vol: Texture3D = Texture3D()     # the confinement's force × dt at the cells

        # ---- 2D composited output (for downstream render layers) ----
        self._output_texture: Texture = Texture()

        # ---- Compute shaders ----
        self._divergence_shader: Divergence3D = Divergence3D()
        self._gradient_shader: Gradient3D = Gradient3D()
        self._jacobi_pressure_shader: JacobiPressure3D = JacobiPressure3D()
        self._sor_pressure_shader: SorPressure3D = SorPressure3D()
        self._multigrid: MultigridPressure = MultigridPressure()
        # The velocity on the cells' faces (FaceVelocity), carried and diffused there; the centers' volume its mean per cell
        self._faces: FaceVelocity = FaceVelocity(swap=True)
        self._faces_scratch: FaceVelocity = FaceVelocity(swap=False)      # MacCormack's predictor, at its first MacCormack step
        self._advect_faces_shader: AdvectFaces3D = AdvectFaces3D()
        self._viscosity_faces_shader: ViscosityFaces3D = ViscosityFaces3D()
        self._faces_to_centers_shader: FacesToCenters3D = FacesToCenters3D()
        self._vorticity_curl_shader: VorticityCurl3D = VorticityCurl3D()
        self._vorticity_force_shader: VorticityForce3D = VorticityForce3D()
        self._buoyancy_shader: Buoyancy3D = Buoyancy3D()
        self._inject_shader: Inject3D = Inject3D()
        self._add_to_faces_shader: AddToFaces3D = AddToFaces3D()     # the forces, the drag and the body force
        self._inject_binary_shader: InjectBinary3D = InjectBinary3D()
        self._add_boolean_shader: AddBoolean = AddBoolean()

        # Bind settings actions
        self.settings.bind(Fluid3DSettings.depth, lambda _: self._request_reallocate())
        reallocate_densities = lambda _: setattr(self, '_densities_pending', True)
        for field in (UpscaledDensitySettings.enabled, UpscaledDensitySettings.width, UpscaledDensitySettings.height,
                      UpscaledDensitySettings.depth):
            self.settings.upscaled.bind(field, reallocate_densities)
        for field in (ParticleSettings.enabled, ParticleSettings.count, ParticleSettings.slices):
            self.settings.particles.bind(field, reallocate_densities)
        self.settings.bind(Fluid3DSettings.width, lambda _: self._request_reallocate())
        self.settings.bind(Fluid3DSettings.height, lambda _: self._request_reallocate())
        mark_wall = lambda _: setattr(self, '_wall_dirty', True)
        self.settings.bind(Fluid3DSettings.wall, mark_wall)

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
        self._guard.allocate()
        self._no_density.allocate(1, 1, 1, GL_RGBA16F)
        self._allocate_simulation_fields()
        self._allocate_densities()

        # 2D obstacle source (output resolution, independent of the grid)
        self._obstacle_source.allocate(self._output_width, self._output_height, GL_R8)

        # 2D composited output
        self._output_texture.allocate(self._output_width, self._output_height, GL_RGBA16F)

        # Allocate shaders
        self._divergence_shader.allocate()
        self._gradient_shader.allocate()
        self._jacobi_pressure_shader.allocate()
        self._sor_pressure_shader.allocate()
        for face_shader in self._face_shaders:
            face_shader.allocate()
        self._vorticity_curl_shader.allocate()
        self._vorticity_force_shader.allocate()
        self._buoyancy_shader.allocate()
        self._inject_shader.allocate()
        self._add_to_faces_shader.allocate()
        self._inject_binary_shader.allocate()
        self._add_boolean_shader.allocate()
        self._over_shader.allocate()
        self._knee_shader.allocate()
        self._light_shader.allocate()
        self._light_blur_shader.allocate()

        self._allocated = True

    def _update_dimensions(self) -> None:
        """Recompute the grids' dimensions from the settings."""
        self._simulation_width = self._align32(self.settings.width)
        self._simulation_height = self._align32(self.settings.height)
        self._aspect = self._simulation_width / self._simulation_height if self._simulation_height > 0 else 1.0
        self._cell = 1.0 / max(1, self._simulation_width)
        self._depth = self.settings.depth
        self._z_extent = self._depth / max(1, self._simulation_width)
        self._depth_scale = self._depth / max(1, min(self._simulation_width, self._simulation_height))

    def _spread(self) -> float:
        """The injection's gaussian sigma as a share of the slab's depth, from `input.spread` in widths."""
        return max(self.settings.input.spread / max(self._z_extent, 1e-6), 0.001)

    @property
    def _z_scale(self) -> float:
        """The body's z to the box's: the box's depth over the real depth it stands for, `input.depth_span`
        times the smaller of its width and height."""
        return self._depth_scale / max(self.settings.input.depth_span, 1e-6)

    @property
    def _densities(self) -> list[DensityCarrier]:
        """The carriers running: the simulation density, then the upscaled and the particles where enabled."""
        densities: list[DensityCarrier] = [self._simulation_density]
        if self.settings.upscaled.enabled:
            densities.append(self._upscaled_density)
        if self.settings.particles.enabled:
            densities.append(self._particle_density)
        return densities

    def _allocate_densities(self) -> None:
        """(Re)allocate the carriers running on the simulation's grid, the others released."""
        for carrier in (self._simulation_density, self._upscaled_density, self._particle_density):
            carrier.deallocate()
        self._particle_texture.deallocate()
        for carrier in self._densities:
            carrier.allocate(self._simulation_width, self._simulation_height, self._depth,
                             self._output_width, self._output_height)
        if self.settings.particles.enabled:
            self._particle_texture.allocate(self._output_width, self._output_height, GL_RGBA16F)
        self._apply_wall_modes()

    def _allocate_simulation_fields(self) -> None:
        """(Re)allocate the simulation's volumes."""
        d = self._depth
        sim_w = self._simulation_width
        sim_h = self._simulation_height

        # Primary simulation fields
        self._velocity.allocate(sim_w, sim_h, d, GL_RGBA16F)
        for faces in (self._faces, self._faces_scratch):
            faces.deallocate()
        self._faces.allocate(sim_w, sim_h, d)
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
        self._vorticity_force_vol.allocate(sim_w, sim_h, d, GL_RGBA16F)
        self._buoyancy_force_vol.allocate(sim_w, sim_h, d, GL_RGBA16F)

        self._apply_wall_modes()

    def deallocate(self) -> None:
        """Release all GPU resources."""
        self._gpu_timer.deallocate()
        for timer in self._pass_timers.values():
            timer.deallocate()
        self._velocity.deallocate()
        self._faces.deallocate()
        self._faces_scratch.deallocate()
        for carrier in (self._simulation_density, self._particle_density, self._upscaled_density):
            carrier.deallocate()
        self._particle_texture.deallocate()
        self._no_density.deallocate()
        self._temperature.deallocate()
        self._pressure.deallocate()
        self._obstacle.deallocate()
        self._obstacle_source.deallocate()
        self._divergence_vol.deallocate()
        self._curl_vol.deallocate()
        self._vorticity_force_vol.deallocate()
        self._buoyancy_force_vol.deallocate()
        self._output_texture.deallocate()
        for scratch in (self._temperature_scratch,):
            scratch.deallocate()

        self._advection.deallocate()
        self._guard.deallocate()
        self._divergence_shader.deallocate()
        self._gradient_shader.deallocate()
        self._jacobi_pressure_shader.deallocate()
        self._sor_pressure_shader.deallocate()
        self._multigrid.deallocate()
        for face_shader in self._face_shaders:
            face_shader.deallocate()
        self._vorticity_curl_shader.deallocate()
        self._vorticity_force_shader.deallocate()
        self._buoyancy_shader.deallocate()
        self._inject_shader.deallocate()
        self._add_to_faces_shader.deallocate()
        self._inject_binary_shader.deallocate()
        self._add_boolean_shader.deallocate()
        self._over_shader.deallocate()
        self._knee_shader.deallocate()
        self._light_shader.deallocate()
        self._light_blur_shader.deallocate()
        for light in (self._light, *self._light_blur):
            light.deallocate()

        self._allocated = False

    def _reload_shaders(self) -> None:
        """Hot-reload all shaders."""
        self._advection.reload()
        for carrier in self._densities:
            carrier.reload()
        self._over_shader.reload()
        self._knee_shader.reload()
        self._light_shader.reload()
        self._light_blur_shader.reload()
        self._divergence_shader.reload()
        self._gradient_shader.reload()
        self._jacobi_pressure_shader.reload()
        self._sor_pressure_shader.reload()
        self._multigrid.reload()
        for face_shader in self._face_shaders:
            face_shader.reload()
        self._vorticity_curl_shader.reload()
        self._vorticity_force_shader.reload()
        self._buoyancy_shader.reload()
        self._inject_shader.reload()
        self._add_to_faces_shader.reload()
        self._inject_binary_shader.reload()

    # ========== Update Pipeline ==========

    def reset(self) -> None:
        """Reset all simulation fields to zero."""
        if not self._allocated:
            return
        self._velocity.clear_all()
        self._faces.clear()
        for carrier in self._densities:
            carrier.reset()
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
        self._timed('faces to centers', self._faces_to_centers)
        self._timed('density', self._advect_density)
        self._timed('light', self._update_light)
        self._timed('composite', self._composite_output)

        self._gpu_timer.end()
        self._guard.update(dt)

    def _timed(self, name: str, work) -> None:
        """`work` inside its pass's GPU timer, `GPU fluid <name>`."""
        timer: GpuTimer = self._pass_timers[name]
        timer.begin()
        work()
        timer.end()

    # ========== Pipeline Steps ==========

    def _add_force_to_velocity(self, force: Texture3D, strength: float = 1.0) -> None:
        """A force at the cells (widths/s²) × `strength` added to the faces, each its two cells' mean."""
        self._add_to_faces_shader.add_force(self._faces.volume, force, strength, self._obstacle, self._closed,
                                            self._has_obstacles)
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

    def _advect_field(self, field: SwapTexture3D, scratch: Texture3D, velocity: Texture3D, step: float,
                      dissipation: float, internal_format: int,
                      evaporation: float = 0.0, measure: int = 0, guard: str = '') -> None:
        """Carry `field` one step along `velocity` by `settings.advection`; its size taken down by `evaporation`,
        read by `measure` (0 a density's alpha, 1 a vector's length, 2 |r|); `guard` the field's name in the guard."""
        if self._advection.carry(self.settings.advection, field, scratch, velocity, self._obstacle, self._aspect,
                                 self._z_extent, step, dissipation, internal_format, self._has_obstacles,
                                 evaporation=evaporation, measure=measure,
                                 guard=self._guard_slot(guard), closed=self._closed, faces=self._faces.volume):
            self._apply_wall_modes()

    @property
    def _cells(self) -> tuple[int, int, int]:
        return (self._simulation_width, self._simulation_height, self._depth)

    def _faces_to_centers(self) -> None:
        """The centers' volume, after the projection: each component the mean of its cell's two faces, which the
        particles, the upscaled density and the monitors read."""
        self._faces_to_centers_shader.use(self._faces.volume, self._velocity, self._obstacle, self._closed,
                                          self._has_obstacles)
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

    def _advect_velocity(self) -> None:
        """The faces carried one step along themselves by `advection`: each traced back on its own; on the result the
        fade, the evaporation, the guard, the top speed and the slip."""
        velocity = self.settings.velocity
        step = self._dt * velocity.advection_scale
        common = dict(obstacle=self._obstacle, cells=self._cells, step=step,
                      closed=self._closed, has_obstacles=self._has_obstacles)
        finish = dict(dissipation=self._calculate_dissipation(self._dt, velocity.fade_time),
                      evaporation=self._evaporation(self._dt, velocity.evaporation_time), max_speed=velocity.max_speed,
                      guard=self._guard_slot('velocity'), wall_keep=self._wall_keep)
        self._faces.swap()
        source = self._faces.back
        if self.settings.advection != Advection.MACCORMACK:
            self._advect_faces_shader.use(SEMI_LAGRANGIAN, source, self._faces.volume, **common, **finish)
        else:
            if not self._faces_scratch.allocated:
                self._faces_scratch.allocate(*self._cells)
            self._advect_faces_shader.use(PREDICTOR, source, self._faces_scratch.volume, **common)
            glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)
            self._advect_faces_shader.use(CORRECTION, source, self._faces.volume, **common, **finish,
                                          predictor=self._faces_scratch.volume)
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

    def _apply_viscosity(self) -> None:
        """The viscosity on the faces: Jacobi iterations of the implicit diffusion, `viscosity_iter` at 60 fps."""
        if self.settings.velocity.viscosity <= 0.0:
            return
        viscosity_dt = self.settings.velocity.viscosity * self._dt       # widths² per second × s
        for iterations in self._viscosity_faces_shader.dispatches(viscosity_iterations(self.settings.velocity.viscosity_iter, self._dt)):
            self._faces.swap()
            self._viscosity_faces_shader.use(self._faces.back, self._faces.volume, self._obstacle,
                                             self._cells, self._cell, viscosity_dt, iterations, self._closed,
                                             self._wall_keep, self._has_obstacles)
            glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

    def _confine_vorticity(self) -> None:
        """Vorticity confinement: the curl, then its force added to the velocity. The curl alone while only the
        upscaled density's turbulence reads it."""
        confining: bool = self.settings.velocity.vorticity > 0.0 and self.settings.velocity.vorticity_radius > 0.0
        if not confining:
            if self._needs_curl:
                self._vorticity_curl_shader.use(self._faces.volume, self._obstacle, self._curl_vol, self._cell, 1.0,
                                                has_obstacles=self._has_obstacles, closed=self._closed, wall_keep=self._wall_keep)
                glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)
            return

        vorticity_radius = self.settings.velocity.vorticity_radius        # cells
        vorticity_force = self.settings.velocity.vorticity * self._cell * self._dt     # epsilon h (N x omega), Fedkiw 2001

        # a. The curl at the cells, from the faces
        self._vorticity_curl_shader.use(
            self._faces.volume,
            self._obstacle,
            self._curl_vol,
            self._cell,
            vorticity_radius,
            has_obstacles=self._has_obstacles,
            closed=self._closed,
            wall_keep=self._wall_keep
        )
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

        # b. The confinement's force × dt at the cells, then onto the faces
        self._vorticity_force_shader.use(self._curl_vol, self._obstacle, self._vorticity_force_vol, self._cell,
                                         vorticity_force, has_obstacles=self._has_obstacles)
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)
        self._add_to_faces_shader.add_force(self._faces.volume, self._vorticity_force_vol, 1.0, self._obstacle,
                                            self._closed, self._has_obstacles)
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

    def _advect_temperature(self) -> None:
        """Advect & dissipate the temperature scalar field."""
        if self.settings.temperature.buoyancy == 0.0:
            if not self._temperature_clear:     # cleared once; nothing adds to it while buoyancy is 0
                self._temperature.clear_all()
                self._temperature_clear = True
            return
        self._temperature_clear = False

        advect_step = self._dt * self.settings.temperature.advection_scale
        dissipation = self._calculate_dissipation(self._dt, self.settings.temperature.fade_time)
        self._advect_field(self._temperature, self._temperature_scratch, self._velocity, advect_step, dissipation, GL_R16F,
                           evaporation=self._evaporation(self._dt, self.settings.temperature.evaporation_time), measure=2,
                           guard='temperature')

    def _apply_buoyancy(self) -> None:
        """Apply buoyancy to velocity: the acceleration f = gravity * (κρ − σ(T − T_ambient)) (widths/s²),
        kept as it is in `buoyancy`, added × dt like every body force (Fedkiw 2001).

        Dense density sinks along gravity; hot (fresh) density rises against it.
        The weight term acts wherever density is, so it stays active even
        with thermal buoyancy at 0.
        """
        temp = self.settings.temperature
        if temp.buoyancy == 0.0 and temp.sink == 0.0:
            return

        sigma = temp.buoyancy
        kappa = temp.sink
        gravity = (temp.gravity_x, temp.gravity_y, temp.gravity_z)

        weight: Texture3D | None = self._simulation_density.weight
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

        self._add_force_to_velocity(self._buoyancy_force_vol, self._dt)

    def _enforce_incompressibility(self) -> None:
        """Pressure projection (divergence → Jacobi solve → gradient subtraction), the three passes on one set
        of walls, closed or open by `wall`."""
        closed: tuple[bool, bool, bool] = self._closed
        # a. The divergence, per cell from its six faces
        self._divergence_shader.use(self._faces.volume, self._obstacle, self._divergence_vol,
                                    self._cell, has_obstacles=self._has_obstacles, closed=closed)
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

        # c. The pressure's gradient subtracted from the faces
        self._faces.swap()
        self._gradient_shader.use(self._faces.back, self._pressure, self._obstacle, self._faces.volume,
                                  self._cell, has_obstacles=self._has_obstacles, closed=closed)
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

    def _advect_density(self) -> None:
        """Move the density one step along the velocity, faded and evaporated, by each carrier running."""
        step = CarrierStep(
            velocity=self._velocity, obstacle=self._obstacle, has_obstacles=self._has_obstacles,
            aspect=self._aspect, z_extent=self._z_extent, dt=self._dt, step=self._dt * self.settings.density.advection_scale,
            dissipation=self._calculate_dissipation(self._dt, self.settings.density.fade_time),
            evaporation=self._evaporation(self._dt, self.settings.density.evaporation_time), advection=self.settings.advection,
            curl=self._curl_vol if self._needs_curl else None, cell=self._cell, guard=self._guard_slot('density'),
            faces=self._faces.volume)
        for carrier in self._densities:
            carrier.update(step)

    def _guard_slot(self, field: str) -> GuardSlot | None:
        """The field's slot in the fluid's guard, None while `guard` is off or no field is named."""
        return self._guard.slot(field) if field and self.settings.guard else None

    @property
    def _closed(self) -> tuple[bool, bool, bool]:
        """Per axis, whether the box's walls are closed (`wall`), as every pass reading beyond the box takes them."""
        return (self.settings.wall == WallMode.CLOSED,) * 3

    @property
    def _face_shaders(self) -> tuple[ComputeShader, ...]:
        return (self._advect_faces_shader, self._viscosity_faces_shader, self._faces_to_centers_shader)

    @property
    def _wall_keep(self) -> float:
        """The velocity's share along a closed wall kept this step: 0.01 over `slip_time` seconds, 1 at 0 (free-slip)."""
        return self._calculate_dissipation(self._dt, self.settings.slip_time)

    @property
    def _needs_curl(self) -> bool:
        """Whether the upscaled density's turbulence reads the curl this step."""
        return self.settings.upscaled.enabled and self.settings.upscaled.turbulence > 0.0

    def _composite_output(self) -> None:
        """The volume into the 2D output, the upscaled amount's where enabled, else the simulation's; the particles
        drawn into their own image and laid over it; then the highlights bent above the knee instead of clipped."""
        volume: DensityCarrier = self._upscaled_density if self.settings.upscaled.enabled else self._simulation_density
        volume.composite(self._output_texture, self._light_now)
        if self.settings.particles.enabled and self._particle_texture.allocated:
            self._particle_density.composite(self._particle_texture)
            self._over_shader.use(self._particle_texture, self._output_texture)
            glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)
        self._knee_shader.use(self._output_texture)
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

    def _update_light(self) -> None:
        """The light this step, for the composite."""
        self._light_now = self._lit()

    def _lit(self) -> CompositeLight | None:
        """The light this step, from the simulation density: each light cell marched toward the light, then
        with `scattering` blurred and mixed in; None while `shadow` is 0, the light released. `opacity` and
        `fog` are over the view, the box's depth, so per width they are over its depth in widths."""
        composite = self.settings.composite
        density: Texture3D | None = self._simulation_density.weight
        if composite.shadow <= 0.0 or density is None:
            for light in (self._light, *self._light_blur):
                light.deallocate()
            return None
        size = (max(1, self._simulation_width // 4), max(1, self._simulation_height // 4), max(1, self._depth // 4))
        for light in (self._light, *self._light_blur):
            if not light.allocated or (light.width, light.height, light.depth) != size:
                light.allocate(*size, GL_RGBA16F)
        box = (1.0, 1.0 / self._aspect, self._z_extent)
        chan = tuple(1.0 + (water - 1.0) * composite.tint for water in WATER_CHANNELS)
        per_width = 1.0 / max(self._z_extent, 1e-6)
        sigma = tuple(view_march(composite.opacity, composite.emission)[0] * per_width * c for c in chan)
        fog = tuple(composite.fog * per_width * c for c in chan)
        point: bool = composite.light == Light.POINT
        vector = self._light_override or (composite.light_x, composite.light_y, composite.light_z)
        self._light_shader.use(density, self._light, box=box, point=point, vector=vector, sigma=sigma, fog=fog)
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)
        light: Texture3D = self._light
        if composite.scattering > 0.0:
            a, b = self._light_blur
            for axis, source, result in ((0, self._light, a), (1, a, b), (2, b, a)):
                self._light_blur_shader.use(source, self._light, result, axis, composite.scattering)
                glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)
            light = a
        self._light_shown = light
        return CompositeLight(light, composite.shadow, composite.anisotropy, point, vector, box)

    def _light_place(self) -> LightPlace | None:
        """Where the light is, in the box's texture coordinates, None while it is off: a lamp at its place, a sun
        where its light, coming in toward the box's center, enters the box."""
        composite = self.settings.composite
        if composite.shadow <= 0.0:
            return None
        vector = self._light_override or (composite.light_x, composite.light_y, composite.light_z)
        if composite.light == Light.POINT:
            return LightPlace(True, (0.5 + 0.5 * vector[0], 0.5 + 0.5 * vector[1], 0.5 + 0.5 * vector[2]))
        length: float = math.sqrt(sum(c * c for c in vector))
        toward = tuple(c / length for c in vector) if length > 1e-6 else (0.0, -1.0, 0.0)   # the light shader's default
        box = (1.0, 1.0 / self._aspect, self._z_extent)
        reach: float = min(0.5 * side / abs(c) for side, c in zip(box, toward) if abs(c) > 1e-9)
        x, y, z = (0.5 + c * reach / side for side, c in zip(box, toward))
        return LightPlace(False, (x, y, z))

    def set_light(self, vector: tuple[float, float, float] | None) -> None:
        """The light's direction toward the sun, or the lamp's place (±1 the box's sides; x right, y down the
        screen, z away from the viewer), over the settings' until None: the app's hook for a light it derives,
        from the pose or elsewhere."""
        self._light_override = vector

    # ========== Deferred ==========

    def _apply_wall_modes(self) -> None:
        """Apply `wall` to every field volume's sides, the velocity's x and y kept at 0 (GL thread only)."""
        wall = _WALL_GL[self.settings.wall]
        for volume in (self._temperature, self._temperature_scratch, self._pressure):
            volume.set_wrap(*wall)
        for volume in (self._velocity, self._temperature, self._pressure, self._temperature_scratch):
            volume.set_wrap_z(*wall)
        for carrier in (self._simulation_density, self._upscaled_density, self._particle_density):
            carrier.set_walls(wall, wall)

    def _handle_deferred_actions(self) -> None:
        """Process reset and reallocation requests queued from the UI thread."""
        if self._reallocate_pending:
            self._reallocate_pending = False
            old_sim = (self._simulation_width, self._simulation_height, self._depth)
            self._update_dimensions()
            if (self._simulation_width, self._simulation_height, self._depth) != old_sim:
                self._allocate_simulation_fields()
                self._densities_pending = True

        if self._densities_pending:
            self._densities_pending = False
            self._allocate_densities()

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
        # the fluid's own time: every pass steps by it, the injections' sources and the drag included, so at
        # time_scale 0.5 the whole fluid runs at half speed and keeps its shapes
        self._dt = self._time_step.update(s.time.time_step, dt, s.time.fixed_fps, s.time.fps_smoothing) * s.time.time_scale

    @staticmethod
    def _align32(v: int) -> int:
        """Round up to the nearest multiple of 32."""
        return (v + 31) & ~31

    @staticmethod
    def _calculate_dissipation(delta_time: float, decay_time: float) -> float:
        """Calculate frame-rate independent decay multiplier.

        Returns pow(0.01, dt / decay_time) -- field reaches 1% after decay_time seconds; 1 at 0, off.
        """
        if decay_time <= 0.0:
            return 1.0
        return pow(0.01, delta_time / decay_time)

    @staticmethod
    def _evaporation(delta_time: float, evaporation_time: float) -> float:
        """The share of a field's full value taken off this step, so it is gone in evaporation_time; 0 off."""
        return delta_time / evaporation_time if evaporation_time > 0.0 else 0.0

    # ========== Input Methods (2D -> 3D injection) ==========

    def drag_velocity(self, texture: Texture, weight: float = 1.0) -> None:
        """The boundary: drag the velocity in the gaussian birth slab toward the body's velocity (x, y on
        the image plane, z toward the camera; widths/s) × input.boundary.velocity_gain, over input.boundary.velocity_drag, where
        the body moves (its speed above DRAG_FLOOR), never past it. The volume's w is -z × `_z_scale`:
        the box's depth stands for `input.depth_span` × the smaller of its width and height of real depth,
        so z is scaled by its depth over that; +W moves fluid away from the viewer (layer 0 is the front face).
        Without depth, z is 0, and so is w.
        """
        if not self._allocated:
            return
        boundary = self.settings.input.boundary
        self._add_to_faces_shader.drag(self._faces.volume, self._cells, texture, self.settings.input.layer, self._spread(),
                                       boundary.velocity_gain, drag_share(self._dt, boundary.velocity_drag), weight,
                                       DRAG_FLOOR, self._z_scale, self._obstacle, self._closed, self._has_obstacles)
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

    def add_acceleration(self, texture: Texture, weight: float = 1.0) -> None:
        """The body force: add the acceleration (x, y on the image plane, z toward the camera; widths/s²)
        × input.body_force.acceleration_gain × dt into the gaussian birth slab; w is -z, as the boundary's."""
        if not self._allocated:
            return
        self._add_to_faces_shader.add_acceleration(self._faces.volume, self._cells, texture, self.settings.input.layer,
                                                   self._spread(), weight * self.settings.input.body_force.acceleration_gain * self._dt,
                                                   self._z_scale, self._obstacle, self._closed, self._has_obstacles)
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

    def add_boundary_density(self, texture: Texture, weight: float = 1.0) -> None:
        """The boundary's dye (RGBA, its color times its presence, the presence as alpha, 0 to 1): the water in the
        slab filled toward it times input.boundary.density_gain (at most fully dyed) over density_drag, never past
        it, by the carriers."""
        boundary = self.settings.input.boundary
        share: float = drag_share(self._dt, boundary.density_drag) * weight
        self._add_density(texture, share, InjectMode.FILL, boundary.density_gain)

    def add_body_force_density(self, texture: Texture, weight: float = 1.0) -> None:
        """The body force's dye (RGBA, a push's color times its presence) at input.body_force.density_rate, fully
        dyed water per second at a full push, × dt into the slab, capped at fully dyed, by the carriers."""
        self._add_density(texture, weight * self.settings.input.body_force.density_rate, InjectMode.RATE)

    def _add_density(self, texture: Texture, gain: float, mode: InjectMode, target_gain: float = 1.0) -> None:
        if not self._allocated:
            return
        for carrier in self._densities:
            carrier.add(texture, gain, self._dt, self.settings.input.layer, self._spread(), mode, target_gain)

    def add_boundary_temperature(self, texture: Texture, weight: float = 1.0) -> None:
        """The boundary's warmth (-1 cold to 1 hot, its presence folded in): the water in the slab dragged toward
        it times input.boundary.temperature_gain (at most ±1) over temperature_drag where the body is."""
        if self.settings.temperature.buoyancy == 0.0:      # nothing reads it
            return
        boundary = self.settings.input.boundary
        share: float = drag_share(self._dt, boundary.temperature_drag) * weight
        self._add_temperature(texture, share, InjectMode.FILL, boundary.temperature_gain)

    def add_body_force_temperature(self, texture: Texture, weight: float = 1.0) -> None:
        """The body force's warmth at input.body_force.temperature_rate, full warmth per second at a full push,
        × dt into the slab, capped at ±1."""
        if self.settings.temperature.buoyancy == 0.0:      # nothing reads it
            return
        self._add_temperature(texture, weight * self.settings.input.body_force.temperature_rate * self._dt, InjectMode.RATE)

    def _add_temperature(self, texture: Texture, strength: float, mode: InjectMode, target_gain: float = 1.0) -> None:
        """The warmth into the temperature's gaussian birth slab: dragged toward (FILL) or added (RATE)."""
        if not self._allocated:
            return
        self._inject_shader.use(texture, self._temperature, self.settings.input.layer, self._spread(), strength,
                                internal_format=GL_R16F, mode=int(mode), scalar=True, gain=target_gain)
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
        obstacle as it comes, and the running carriers' (the simulation's `density`, the `upscaled_density`)."""
        extent: float = self._z_extent
        light: LightPlace | None = self._light_place()

        def volume(texture: Texture3D, kind: TextureKind) -> Port:
            return Port(texture, kind, extent, light=light)

        return {'obstacle_source': Port(self._obstacle_source, TextureKind.SCALAR),
                'obstacle': volume(self._obstacle, TextureKind.SCALAR),
                'velocity': volume(self._velocity, TextureKind.VELOCITY),
                'velocity_faces': volume(self._faces.volume, TextureKind.VELOCITY),
                'vorticity': volume(self._curl_vol, TextureKind.VELOCITY),
                'temperature': volume(self._temperature, TextureKind.SCALAR),
                'buoyancy': volume(self._buoyancy_force_vol, TextureKind.ACCELERATION),
                'divergence': volume(self._divergence_vol, TextureKind.SCALAR),
                'pressure': volume(self._pressure, TextureKind.SCALAR),
                **({'light': Port(self._light_shown, TextureKind.IMAGE, extent, held=3, light=light)} if self._light_shown.allocated else {}),
                **{name: replace(port, light=light) if port.volume else port
                   for carrier in self._densities for name, port in carrier.ports.items()}}
