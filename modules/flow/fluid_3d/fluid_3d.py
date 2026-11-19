"""Fluid3D - 3D Navier-Stokes fluid simulation on the GPU: the velocity on the cells' faces, the temperature, the
pressure and the obstacle on one grid, the density carried by `Density` and, where enabled, `FineDensity` and
`ParticleDensity`; the bridge's two systems go in by `drag_velocity`, `add_acceleration`, `add_density`
and `add_temperature`, the expansion by `add_expansion`, each at its pixels' z positions; `update` steps it and composites the look. How and why:
modules/flow/FLUID_3D.md, "The fluid", "The inputs" and "The walls".
"""
import math
from dataclasses import replace

from OpenGL.GL import GL_CLAMP_TO_BORDER, GL_CLAMP_TO_EDGE, GL_NEAREST, GL_REPEAT, GL_R16F, GL_R32F, GL_R8, GL_RGBA16F, GL_SHADER_IMAGE_ACCESS_BARRIER_BIT, GL_TEXTURE_FETCH_BARRIER_BIT, glMemoryBarrier

from modules.gl import ComputeShader, StorageBuffer, SwapFbo, Texture, Texture3D, SwapTexture3D, GpuTimer
from ..wall_mode import WallMode
from .. import pressure_reach
from ..density_blend import DensityBlend
from ..drag import DRAG_FLOOR, drag_share
from .fluid_settings import Advection, Light, Limiter, PressureSolver, ScatteringModel, Fluid3DSettings, PressureSettings, ParticleSettings, DensitySettings, FineDensitySettings
from .multigrid_pressure import MultigridPressure
from .face_velocity import FaceVelocity
from ..shaders import AddBoolean
from ..flow_util import FlowUtil
from ..port import LightPlace, Port, TextureKind
from ..time_step import REFERENCE_STEP, TimeStep, viscosity_iterations
from ..finite_guard import FiniteGuard, GuardSlot
from .density_carrier import CarrierStep, DensityCarrier
from .fine_density import FineDensity
from .field_advection import FieldAdvection
from .density import Density
from .particle_density import ParticleDensity
from .shaders.face_binding import Cell
from .shaders import (AdvectFaces3D, FacesToCenters3D, ViscosityFaces3D, SEMI_LAGRANGIAN, PREDICTOR, CORRECTION,
                      
    Divergence3D, ExpansionMean3D, PARTIAL_BYTES, Gradient3D,
    JacobiPressure3D, SorPressure3D,
    VorticityCurl3D, VorticityForce3D, Buoyancy3D, AddToFaces3D,
    Inject3D,
    CompositeOver, Knee,
    CompositeLight, LightVolume3D, LightSweep3D, LightBlur3D, CausticNet, LayerLight3D, IDENTITY, MarchLook, march_look, half_extinction, fog_extinction, sweep_map, sweep_size, InjectMode,
    InjectBinary3D
)


# Combined memory barrier bits (cast to int for Pylance compatibility)
_BARRIER_FETCH_AND_IMAGE: int = int(GL_TEXTURE_FETCH_BARRIER_BIT) | int(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
_BARRIER_IMAGE: int = int(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

_WALL_GL: dict = {
    WallMode.OPEN:   (GL_CLAMP_TO_BORDER, (0.0, 0.0, 0.0, 0.0)),
    WallMode.CLOSED: (GL_CLAMP_TO_EDGE,   (0.0, 0.0, 0.0, 0.0)),
}

FLUID_PASSES: tuple[str, ...] = ('advect velocity', 'viscosity', 'vorticity', 'buoyancy', 'pressure', 'faces to centers',
                                 'advect temperature', 'density', 'light', 'composite')     # a step's passes, each a GPU timer
FULL_WIDTH_HALF_MAXIMUM: float = 2.0 * math.sqrt(2.0 * math.log(2.0))   # a gaussian's width at half its height, in sigmas
CAUSTIC_NET_SIZE: int = 256   # the caustic net's texels across its tile

import logging
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 3D Fluid Simulation
# ---------------------------------------------------------------------------

class Fluid3D:
    """3D Navier-Stokes fluid simulation, every field a GL_TEXTURE_3D on one grid.

    Fields:
        faces:       FaceVelocity   RGBA16F  the velocity on the cells' faces, a face longer on every axis
        velocity:    SwapTexture3D  RGBA16F  (u, v, w, spare) the faces averaged to the cells' centers
        density:     Density, with FineDensity and ParticleDensity where enabled
        temperature: SwapTexture3D  R16F
        pressure:    SwapTexture3D  R32F
        obstacle:    Texture3D      R8

    Intermediate (single-buffer):
        divergence:       Texture3D R16F
        expansion:        Texture3D R16F     a presence, summed over a tick's injections
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
        self._cell: Cell = (1.0, 1.0, 1.0)   # a cell's sides in widths, x, y and z: 1 / the grid's width across, stretched in z

        self._dt: float = REFERENCE_STEP
        self._time_step: TimeStep = TimeStep()

        # The slab's thickness in width units: cubic cells, so layers / width (advection: uRdx.z = 1/z_extent)
        self._z_extent: float = 1.0

        self._has_obstacles: bool = False

        self._allocated: bool = False

        self._reallocate_pending: bool = False
        self._densities_pending: bool = False
        self._densities_clear: bool = False     # the density's color space changed: what it holds reads wrong
        self._wall_dirty: bool = False

        self._gpu_timer: GpuTimer = GpuTimer("GPU fluid")
        self._pass_timers: dict[str, GpuTimer] = {name: GpuTimer(f"GPU fluid {name}") for name in FLUID_PASSES}

        # ---- Volumetric fields (SwapTexture3D for ping-pong) ----
        # Velocity always uses Dirichlet (v=0 at walls); others configurable via _apply_wall_modes().
        self._velocity: SwapTexture3D = SwapTexture3D(wrap=GL_CLAMP_TO_BORDER, border_color=(0.0, 0.0, 0.0, 0.0))
        self._temperature: SwapTexture3D = SwapTexture3D()
        self._temperature_clear: bool = False      # both buffers 0 since the temperature went off
        self._pressure: SwapTexture3D = SwapTexture3D()
        # The MacCormack predictors, one per carried field, allocated at its first MacCormack step
        self._temperature_scratch: Texture3D = Texture3D()
        # The density's carriers: the density always, the fine density and the particles by their `enabled`;
        # the particles drawn into their own image, then over the volume
        self._advection: FieldAdvection = FieldAdvection()
        # The fields' non-finite values, set to 0 where each is advected, counted and reported
        self._guard: FiniteGuard = FiniteGuard('fluid', ('velocity', 'density', 'temperature'))
        self._density: Density = Density(self.settings, self._advection)
        self._particle_density: ParticleDensity = ParticleDensity(self.settings)
        self._fine_density: FineDensity = FineDensity(self.settings, self._advection, self._density)
        self._particle_texture: Texture = Texture()
        self._over_shader: CompositeOver = CompositeOver()
        self._knee_shader: Knee = Knee()               # the output's highlights bent, not clipped
        # The light: how much reaches each cell of a grid at `light.grid`'s share of the simulation's size, and its blur
        # for the scattering; allocated while `composite.light` is enabled
        self._light: Texture3D = Texture3D(wrap=GL_CLAMP_TO_EDGE)
        self._light_blur: tuple[Texture3D, Texture3D] = (Texture3D(wrap=GL_CLAMP_TO_EDGE), Texture3D(wrap=GL_CLAMP_TO_EDGE))
        self._light_shader: LightVolume3D = LightVolume3D()        # a lamp's: each cell marched to it
        self._light_sweep_shader: LightSweep3D = LightSweep3D()    # a sun's: one sweep of parallel lines
        self._light_blur_shader: LightBlur3D = LightBlur3D()
        self._light_override: tuple[float, float, float] | None = None     # set_light's, over the settings
        self._light_shown: Texture3D = self._light      # the light the composite read last: direct, or scattered
        self._light_now: CompositeLight | None = None   # this step's light, `_update_light`'s, for the composite
        # The caustic net, drawn each step while `caustics` is on, and the water's light a layer of the march at a
        # time while `fog_scattering` is on; the clock the net moves by
        self._caustic_net: Texture = Texture(wrap=GL_REPEAT)
        self._caustic_shader: CausticNet = CausticNet()
        # The light per column of the light grid and layer of the march, what the composite reads: on the dye (the
        # sun's and the sky's) and the water's (the rays, while `fog_scattering` is on)
        self._direct: Texture3D = Texture3D(wrap=GL_CLAMP_TO_EDGE)
        self._rays: Texture3D = Texture3D(wrap=GL_CLAMP_TO_EDGE)
        self._layer_light_shader: LayerLight3D = LayerLight3D()
        self._light_clock: float = 0.0
        self._sky: Texture3D = Texture3D(wrap=GL_CLAMP_TO_EDGE)     # the light from straight above, while `sky_fill` is on
        self._no_density: Texture3D = Texture3D()
        # Obstacle: CLAMP_TO_BORDER(1) = out-of-bounds = obstacle
        self._obstacle: Texture3D = Texture3D(interpolation=GL_NEAREST, wrap=GL_CLAMP_TO_BORDER, border_color=(1.0, 0.0, 0.0, 0.0))
        # 2D obstacle source (authoritative, never affected by a reallocation)
        self._obstacle_source: SwapFbo = SwapFbo(interpolation=GL_NEAREST, wrap=GL_CLAMP_TO_BORDER, border_color=(1.0, 0.0, 0.0, 0.0))

        # ---- Intermediate volumes (single buffer, no ping-pong) ----
        self._divergence_vol: Texture3D = Texture3D()
        # The expansion, a presence summed over this tick's injections, and in a closed box its mean over the
        # water's cells, per workgroup and in all; fresh from its first injection to the projection
        self._expansion_vol: Texture3D = Texture3D()
        self._expansion_partials: StorageBuffer = StorageBuffer()
        self._expansion_mean: StorageBuffer = StorageBuffer()
        self._expansion_fresh: bool = False
        # The confinements' curls: the first running one's (`_curl_vol`, which the turbulence reads too), then one
        # for each further confinement, allocated when it first runs
        self._curl_vol: Texture3D = Texture3D(wrap=GL_CLAMP_TO_BORDER, border_color=(0.0, 0.0, 0.0, 0.0))
        self._more_curls: list[Texture3D] = [Texture3D(wrap=GL_CLAMP_TO_BORDER, border_color=(0.0, 0.0, 0.0, 0.0))
                                             for _ in range(VorticityForce3D.MAX_CONFINEMENTS - 1)]
        self._buoyancy_force_vol: Texture3D = Texture3D()       # the buoyancy at the cells, for its monitor
        self._buoyancy_shown: bool = False      # a monitor read the buoyancy since the last step
        self._vorticity_force_vol: Texture3D = Texture3D()     # the confinements' force × dt at the cells

        # ---- 2D composited output (for downstream render layers) ----
        self._output_texture: Texture = Texture()

        # ---- Compute shaders ----
        self._divergence_shader: Divergence3D = Divergence3D()
        self._expansion_mean_shader: ExpansionMean3D = ExpansionMean3D()
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
        self.settings.bind(Fluid3DSettings.depth_ratio, lambda _: self._request_reallocate())
        reallocate_densities = lambda _: setattr(self, '_densities_pending', True)
        for field in (FineDensitySettings.enabled, FineDensitySettings.width, FineDensitySettings.height,
                      FineDensitySettings.depth):
            self.settings.fine_density.bind(field, reallocate_densities)
        for field in (DensitySettings.width, DensitySettings.height, DensitySettings.depth):
            self.settings.density.bind(field, reallocate_densities)
        self.settings.density.bind(DensitySettings.color_space, lambda _: setattr(self, '_densities_clear', True))
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
        self._expansion_mean_shader.allocate()
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
        self._light_sweep_shader.allocate()
        self._caustic_shader.allocate()
        self._layer_light_shader.allocate()
        self._light_blur_shader.allocate()

        self._allocated = True

    def _update_dimensions(self) -> None:
        """Recompute the grids' dimensions from the settings."""
        self._simulation_width = self._align32(self.settings.width)
        self._simulation_height = self._align32(self.settings.height)
        self._aspect = self._simulation_width / self._simulation_height if self._simulation_height > 0 else 1.0
        self._depth = self.settings.depth
        self._z_extent = self.settings.depth_ratio      # the box's depth in widths
        across: float = 1.0 / max(1, self._simulation_width)
        self._cell = (across, across, self._z_extent / max(1, self._depth))
        self.settings.z_stretch = self._cell[2] / across

    def _spread(self) -> float:
        """The injection's gaussian sigma as a share of the box's depth, from `input.spread`, the body's thickness
        in widths where it is at least half there (its full width at half maximum, 2.355 sigma)."""
        return self.settings.input.spread / FULL_WIDTH_HALF_MAXIMUM / max(self._z_extent, 1e-6)

    @property
    def _densities(self) -> list[DensityCarrier]:
        """The carriers running: the density, then the fine density and the particles where enabled."""
        densities: list[DensityCarrier] = [self._density]
        if self.settings.fine_density.enabled:
            densities.append(self._fine_density)
        if self.settings.particles.enabled:
            densities.append(self._particle_density)
        return densities

    def _allocate_densities(self) -> None:
        """(Re)allocate the carriers running, each at its own size, the others released."""
        for carrier in (self._density, self._fine_density, self._particle_density):
            carrier.deallocate()
        self._particle_texture.deallocate()
        for carrier in self._densities:
            carrier.allocate(self._simulation_width, self._simulation_height, self._depth, self._z_extent,
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
        self._multigrid.allocate(sim_w, sim_h, d, self._cell)
        self._obstacle.allocate(sim_w, sim_h, d, GL_R8)
        if self._has_obstacles:
            self._inject_binary_shader.use(self._obstacle_source, self._obstacle, mode=1)
            glMemoryBarrier(_BARRIER_IMAGE)

        # Intermediate volumes
        self._divergence_vol.allocate(sim_w, sim_h, d, GL_R16F)
        self._expansion_vol.allocate(sim_w, sim_h, d, GL_R16F)
        self._expansion_vol.clear()
        self._expansion_fresh = False
        self._expansion_partials.allocate(ExpansionMean3D.groups((sim_w, sim_h, d)) * PARTIAL_BYTES)
        self._expansion_mean.allocate(4)
        self._curl_vol.allocate(sim_w, sim_h, d, GL_RGBA16F)
        self._buoyancy_force_vol.allocate(sim_w, sim_h, d, GL_RGBA16F)
        self._vorticity_force_vol.allocate(sim_w, sim_h, d, GL_RGBA16F)

        self._apply_wall_modes()

    def deallocate(self) -> None:
        """Release all GPU resources."""
        self._gpu_timer.deallocate()
        for timer in self._pass_timers.values():
            timer.deallocate()
        self._velocity.deallocate()
        self._faces.deallocate()
        self._faces_scratch.deallocate()
        for carrier in (self._density, self._particle_density, self._fine_density):
            carrier.deallocate()
        self._particle_texture.deallocate()
        self._no_density.deallocate()
        self._temperature.deallocate()
        self._pressure.deallocate()
        self._obstacle.deallocate()
        self._obstacle_source.deallocate()
        self._divergence_vol.deallocate()
        self._expansion_vol.deallocate()
        self._expansion_partials.deallocate()
        self._expansion_mean.deallocate()
        self._curl_vol.deallocate()
        for curl in self._more_curls:
            curl.deallocate()
        self._buoyancy_force_vol.deallocate()
        self._vorticity_force_vol.deallocate()
        self._output_texture.deallocate()
        for scratch in (self._temperature_scratch,):
            scratch.deallocate()

        self._advection.deallocate()
        self._guard.deallocate()
        self._divergence_shader.deallocate()
        self._expansion_mean_shader.deallocate()
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
        self._light_sweep_shader.deallocate()
        self._caustic_shader.deallocate()
        self._layer_light_shader.deallocate()
        self._light_blur_shader.deallocate()
        for light in self._light_volumes:
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
        self._light_sweep_shader.reload()
        self._caustic_shader.reload()
        self._layer_light_shader.reload()
        self._light_blur_shader.reload()
        self._divergence_shader.reload()
        self._expansion_mean_shader.reload()
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
        self._timed('buoyancy', self._apply_buoyancy)
        self._timed('pressure', self._enforce_incompressibility)
        self._timed('faces to centers', self._faces_to_centers)
        self._timed('advect temperature', self._advect_temperature)
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

    def _advect_field(self, field: SwapTexture3D, scratch: Texture3D, velocity: Texture3D, advection: Advection,
                      step: float, dissipation: float, internal_format: int,
                      evaporation: float = 0.0, evaporation_knee: float = 0.1, measure: int = 0, guard: str = '',
                      limiter: Limiter = Limiter.CLAMP) -> None:
        """Carry `field` one step along `velocity` by `advection`; its size taken down by the saturating loss
        `evaporation` with its knee, read by `measure` (0 a density's alpha, 1 a vector's length, 2 |r|); `guard` the
        field's name in the guard."""
        if self._advection.carry(advection, field, scratch, velocity, self._obstacle, self._aspect,
                                 self._z_extent, step, dissipation, internal_format, self._has_obstacles,
                                 evaporation=evaporation, evaporation_knee=evaporation_knee, measure=measure,
                                 guard=self._guard_slot(guard), closed=self._closed, faces=self._faces.volume, limiter=limiter):
            self._apply_wall_modes()

    @property
    def _cells(self) -> tuple[int, int, int]:
        return (self._simulation_width, self._simulation_height, self._depth)

    def _faces_to_centers(self) -> None:
        """The centers' volume, after the projection: each component the mean of its cell's two faces, which the
        particles, the densities and the monitors read."""
        self._faces_to_centers_shader.use(self._faces.volume, self._velocity, self._obstacle, self._closed,
                                          self._has_obstacles)
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

    def _advect_velocity(self) -> None:
        """The faces carried one step along themselves by `velocity.advection`: each traced back on its own; on the result the
        fade, the evaporation, the guard, the top speed and the slip."""
        velocity = self.settings.velocity
        step = self._dt * velocity.advection_scale
        common = dict(obstacle=self._obstacle, cells=self._cells, cell=self._cell, step=step,
                      closed=self._closed, has_obstacles=self._has_obstacles)
        finish = dict(dissipation=self._calculate_dissipation(self._dt, velocity.fade_time),
                      evaporation=self._evaporation(self._dt, velocity.evaporation_time),
                      evaporation_knee=velocity.evaporation_knee,
                      guard=self._guard_slot('velocity'), wall_keep=self._wall_keep)
        self._faces.swap()
        source = self._faces.back
        if self.settings.velocity.advection != Advection.MACCORMACK:
            self._advect_faces_shader.use(SEMI_LAGRANGIAN, source, self._faces.volume, **common, **finish)
        else:
            if not self._faces_scratch.allocated:
                self._faces_scratch.allocate(*self._cells)
            self._advect_faces_shader.use(PREDICTOR, source, self._faces_scratch.volume, **common)
            glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)
            self._advect_faces_shader.use(CORRECTION, source, self._faces.volume, **common, **finish,
                                          predictor=self._faces_scratch.volume, clamp=velocity.limiter == Limiter.CLAMP)
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

    def _apply_viscosity(self) -> None:
        """The viscosity on the faces: Jacobi iterations of the implicit diffusion, `viscosity.iterations` at 60 fps;
        while `viscosity.enabled` is on and the viscosity above 0."""
        viscosity = self.settings.viscosity
        if not viscosity.enabled or viscosity.viscosity <= 0.0:
            return
        viscosity_dt = viscosity.viscosity * self._dt       # widths² per second × s
        for iterations in self._viscosity_faces_shader.dispatches(viscosity_iterations(viscosity.iterations, self._dt)):
            self._faces.swap()
            self._viscosity_faces_shader.use(self._faces.back, self._faces.volume, self._obstacle,
                                             self._cells, self._cell, viscosity_dt, iterations, self._closed,
                                             self._wall_keep, self._has_obstacles)
            glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

    def _confine_vorticity(self) -> None:
        """Vorticity confinement, each of the three (`vorticity.strength_1` to `_3` at `vorticity.radius_1` to `_3`) keeping swirls of its own
        size alive: every curl in one pass, their forces summed at the cells in one, added to the faces once, each
        face its two cells' mean, while `vorticity.enabled` is on, each confinement while its strength is above 0. The
        first running confinement's curl is the one the fine density's turbulence reads; the curl alone while only it
        does."""
        vorticity = self.settings.vorticity
        running: list[tuple[float, float]] = [(strength, radius) for strength, radius in
                                              ((vorticity.strength_1, vorticity.radius_1),
                                               (vorticity.strength_2, vorticity.radius_2),
                                               (vorticity.strength_3, vorticity.radius_3))
                                              if vorticity.enabled and strength > 0.0 and radius > 0.0]
        if not running:
            if self._needs_curl:
                self._vorticity_curl_shader.use(self._faces.volume, self._obstacle, [self._curl_vol], self._cell, [self._cell[0]],
                                                has_obstacles=self._has_obstacles, closed=self._closed, wall_keep=self._wall_keep)
                glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)
            return
        size = (self._curl_vol.width, self._curl_vol.height, self._curl_vol.depth)
        curls: list[Texture3D] = [self._curl_vol]
        for curl in self._more_curls[:len(running) - 1]:
            if (curl.width, curl.height, curl.depth) != size:
                curl.deallocate()
                curl.allocate(*size, GL_RGBA16F)        # once, when it first runs
            curls.append(curl)

        # a. The curls at the cells, from the faces, in one pass
        self._vorticity_curl_shader.use(self._faces.volume, self._obstacle, curls, self._cell, [radius for _, radius in running],
                                        has_obstacles=self._has_obstacles, closed=self._closed, wall_keep=self._wall_keep)
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

        # b. Their force × dt at the cells, epsilon h (N x omega) each (Fedkiw 2001; h the xy cell), summed, then onto the faces
        h_dt: float = self._cell[0] * self._dt
        self._vorticity_force_shader.use(curls, self._obstacle, self._vorticity_force_vol, self._cell,
                                         [strength * h_dt for strength, _ in running], has_obstacles=self._has_obstacles)
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)
        self._add_to_faces_shader.add_force(self._faces.volume, self._vorticity_force_vol, 1.0, self._obstacle,
                                            self._closed, self._has_obstacles)
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

    @property
    def _expansion_rate(self) -> float:
        """The expansion's rate (1/s at full presence) while `expansion.enabled` is on, else 0."""
        expansion = self.settings.expansion
        return expansion.rate if expansion.enabled else 0.0

    @property
    def _temperature_runs(self) -> bool:
        """Whether the temperature is injected, carried and lifts: `temperature.enabled` and its `buoyancy` above 0."""
        temperature = self.settings.temperature
        return temperature.enabled and temperature.buoyancy > 0.0

    def _advect_temperature(self) -> None:
        """The temperature carried one step by the projected faces, beside the density and as it is (the density's
        `advection`, `limiter` and `advection_scale`), faded and evaporated by its own; the buoyancy reads it the next
        step; while it does not run, cleared once and skipped."""
        if not self._temperature_runs:
            if not self._temperature_clear:     # cleared once; nothing adds to it while it does not run
                self._temperature.clear_all()
                self._temperature_clear = True
            return
        self._temperature_clear = False

        temperature = self.settings.temperature
        density = self.settings.density
        advect_step = self._dt * density.advection_scale
        dissipation = self._calculate_dissipation(self._dt, temperature.fade_time)
        self._advect_field(self._temperature, self._temperature_scratch, self._velocity, density.advection, advect_step, dissipation, GL_R16F,
                           evaporation=self._evaporation(self._dt, temperature.evaporation_time),
                           evaporation_knee=temperature.evaporation_knee, measure=2,
                           guard='temperature', limiter=density.limiter)

    def _apply_buoyancy(self) -> None:
        """Apply buoyancy to velocity: the acceleration f = gravity * (κρ − σT) (widths/s²), ρ the density's
        amount, T the warmth, the difference from the room's (Boussinesq), kept as it is in `buoyancy`, added × dt
        like every body force (Fedkiw 2001).

        Dense density sinks along gravity (`density.sink`); warm water rises against it (`temperature.buoyancy`,
        while the temperature runs). The pass runs while either is above 0 and `gravity` is not 0: one
        dispatch adds it straight to the faces; the force at the cells is made only for its monitor.
        """
        shown: bool = self._buoyancy_shown
        self._buoyancy_shown = False
        sigma: float = self.settings.temperature.buoyancy if self._temperature_runs else 0.0
        kappa: float = self.settings.density.sink
        g = self.settings.gravity
        gravity: tuple[float, float, float] = (g.x, g.y, g.z)
        if (sigma <= 0.0 and kappa <= 0.0) or gravity == (0.0, 0.0, 0.0):
            return

        weight: Texture3D | None = self._density.weight
        density: Texture3D = weight if weight is not None else self._no_density
        self._add_to_faces_shader.add_buoyancy(self._faces.volume, self._cells, self._temperature, density,
                                               sigma, kappa, gravity, self._dt,
                                               self._obstacle, self._closed, self._has_obstacles)
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)
        if shown:
            self._buoyancy_shader.use(self._temperature, density, self._obstacle, self._buoyancy_force_vol,
                                      sigma, kappa, gravity=gravity, has_obstacles=self._has_obstacles)
            glMemoryBarrier(_BARRIER_IMAGE)

    def _show_buoyancy(self) -> None:
        """A monitor reads the buoyancy: the next step makes it at the cells."""
        self._buoyancy_shown = True

    def _enforce_incompressibility(self) -> None:
        """Pressure projection (divergence → Jacobi solve → gradient subtraction), the three passes on one set
        of walls, closed or open by `wall`; with this tick's expansion the solve makes the divergence φ, not 0; the
        screened equation (∇² − 1/reach²) p = div while `pressure.reach` is under 1 (`pressure_reach.screening`, the
        box's longest side its scale); with `pressure.enabled` off no pressure, all three skipped."""
        share: float = self.settings.pressure.reach
        if not self.settings.pressure.enabled or share <= 0.0:
            self._expansion_fresh = False
            return
        closed: tuple[bool, bool, bool] = self._closed
        screening: float = pressure_reach.screening(share, max(1.0, 1.0 / self._aspect, self._z_extent))
        # a. The divergence, per cell from its six faces, less φ; in a closed box at reach 1 less φ's mean first, as the
        # solve then needs the divergence to sum to 0
        expansion: tuple[Texture3D, float, StorageBuffer | None] | None = None
        rate: float = self._expansion_rate
        if self._expansion_fresh and rate != 0.0:
            mean: StorageBuffer | None = None
            if all(closed) and screening == 0.0:
                self._expansion_mean_shader.use(self._expansion_vol, self._obstacle, self._has_obstacles,
                                                self._expansion_partials, self._expansion_mean)
                mean = self._expansion_mean
            expansion = (self._expansion_vol, rate, mean)
        self._expansion_fresh = False
        self._divergence_shader.use(self._faces.volume, self._obstacle, self._divergence_vol,
                                    self._cell, has_obstacles=self._has_obstacles, closed=closed, expansion=expansion)
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

        # b. Solve Poisson equation for pressure, by the solver picked
        pressure: PressureSettings = self.settings.pressure
        iterations = pressure.iterations     # the grid sets what it needs, not the step
        if pressure.solver == PressureSolver.MULTIGRID:
            self._multigrid.solve(self._pressure, self._divergence_vol, self._obstacle, self._cell, pressure.cycles,
                                  has_obstacles=self._has_obstacles, closed=closed, screening=screening)
        elif pressure.solver == PressureSolver.SOR:
            self._sor_pressure_shader.solve(self._pressure, self._divergence_vol, self._obstacle, self._cell,
                                            iterations, pressure.relaxation,
                                            has_obstacles=self._has_obstacles, closed=closed, screening=screening)
        else:
            result = self._jacobi_pressure_shader.solve(
                self._pressure,
                self._pressure.back,
                self._divergence_vol,
                self._obstacle,
                self._cell,
                total_iterations=iterations,
                iterations_per_dispatch=5,
                has_obstacles=self._has_obstacles, closed=closed, screening=screening
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
            evaporation=self._evaporation(self._dt, self.settings.density.evaporation_time),
            evaporation_knee=self.settings.density.evaporation_knee, advection=self.settings.density.advection,
            curl=self._curl_vol if self._needs_curl else None, cell=self._cell[0], guard=self._guard_slot('density'),
            limiter=self.settings.density.limiter)
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
        """Whether the fine density's turbulence reads the curl this step."""
        return self.settings.fine_density.enabled and self.settings.fine_density.turbulence > 0.0

    def _composite_output(self) -> None:
        """The volume into the 2D output, the fine density where enabled, else the density; the particles
        drawn into their own image and laid over it; then the highlights bent above the knee instead of clipped."""
        volume: DensityCarrier = self._fine_density if self.settings.fine_density.enabled else self._density
        look: MarchLook = self._march_look()
        volume.composite(self._output_texture, look, self._light_now)
        if self.settings.particles.enabled and self._particle_texture.allocated:
            self._particle_density.composite(self._particle_texture, look)
            self._over_shader.use(self._particle_texture, self._output_texture)
            glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)
        self._knee_shader.use(self._output_texture)
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

    @property
    def _light_volumes(self) -> tuple[Texture | Texture3D, ...]:
        """Everything the light allocates, released while it is off."""
        return (self._light, *self._light_blur, self._sky, self._caustic_net, self._direct, self._rays)

    def _march_look(self) -> MarchLook:
        """The composite's terms over the view, the box's depth in widths, from the panel's."""
        composite = self.settings.composite
        tint = composite.tint
        return march_look(composite.emission, composite.opacity, composite.fog, (tint.r, tint.g, tint.b), self._z_extent,
                          composite.clear_below)

    def _update_light(self) -> None:
        """The light this step, for the composite: the light grid, the caustic net and the light per layer."""
        self._light_clock += self._dt
        light: CompositeLight | None = self._lit()
        if light is not None:
            light = self._with_caustics(light)
            light = self._layer_light(light)
        self._light_now = light

    def _with_caustics(self, light: CompositeLight) -> CompositeLight:
        """The light with the caustic net drawn at this moment, while `caustics` is on; released while off."""
        composite = self.settings.composite.light
        if composite.caustics <= 0.0:
            self._caustic_net.deallocate()
            return light
        if not self._caustic_net.allocated:
            self._caustic_net.allocate(CAUSTIC_NET_SIZE, CAUSTIC_NET_SIZE, GL_R16F)
        self._caustic_shader.use(self._caustic_net, self._light_clock * composite.caustic_speed)
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)
        return replace(light, caustics=composite.caustics, caustic_size=composite.caustic_size,
                       caustic_depth=composite.caustic_depth, net=self._caustic_net)

    def _layer_light(self, light: CompositeLight) -> CompositeLight:
        """The light with what the composite reads, per column of the light grid and layer of the composited volume:
        the light on the dye, the sky's in it while `sky_fill` is on, and the water's while `fog_scattering` and
        `fog` are on, released while off."""
        composite = self.settings.composite
        layers: int = self.settings.fine_density.depth if self.settings.fine_density.enabled else self.settings.density.depth
        size = (light.grid[0], light.grid[1], max(1, layers))
        lit = composite.light
        has_sky: bool = lit.sky_fill > 0.0
        has_rays: bool = lit.fog_scattering > 0.0
        for volume, on in ((self._direct, True), (self._rays, has_rays)):
            if not on:
                volume.deallocate()
            elif not volume.allocated or (volume.width, volume.height, volume.depth) != size:
                volume.allocate(*size, GL_RGBA16F)
        sky, sky_map = self._lit_sky() if has_sky else (None, IDENTITY)
        if not has_sky:
            self._sky.deallocate()
        fog = self._march_look().fog
        octaves: float = lit.scattering if lit.scattering_model == ScatteringModel.OCTAVES else 0.0
        self._layer_light_shader.use(light, sky, sky_map, self._direct, self._rays if has_rays else None,
                                     octaves=octaves, powder=lit.powder,
                                     sky_fill=lit.sky_fill,
                                     fog=fog, fog_grey=half_extinction(composite.fog) * self._z_extent,
                                     scatter=lit.fog_scattering * lit.intensity,
                                     sub_steps=max(2, 2 * -(-light.grid[2] // size[2])))   # two a light cell the layer spans
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)
        return replace(light, direct=self._direct, rays=self._rays if has_rays else None)

    def _light_extinction(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """The light's extinction per width: the density's per unit of amount, grey, from its half-thickness
        `opacity`, and the water's per channel, from its half-distance `fog` and `tint`."""
        composite = self.settings.composite
        density: float = half_extinction(composite.opacity)
        tint = composite.tint
        return (density, density, density), fog_extinction(composite.fog, (tint.r, tint.g, tint.b))

    def _lit(self) -> CompositeLight | None:
        """The light this step, from the density: a sun's swept through the light grid in parallel lines,
        each light cell marched toward a lamp; then with `scattering` blurred and mixed in; None while
        the light is off, the light released."""
        composite = self.settings.composite
        density: Texture3D | None = self._density.weight
        lit = composite.light
        if not lit.enabled or density is None:
            for light in self._light_volumes:
                light.deallocate()
            return None
        share: int = int(lit.grid)
        size = (max(1, self._simulation_width // share), max(1, self._simulation_height // share), max(1, self._depth // share))
        point: bool = lit.source == Light.POINT
        vector = self._light_override or (lit.x, lit.y, lit.z)
        length: float = math.sqrt(sum(c * c for c in vector))
        toward = (vector[0] / length, vector[1] / length, vector[2] / length) if length > 1e-6 else (0.0, -1.0, 0.0)
        light_cell = (self._cell[0] * share, self._cell[1] * share, self._z_extent / size[2])
        stored = size if point else sweep_size(size, toward, light_cell)     # a sun's grid is stored sheared, a line a column
        light_map: tuple[float, ...] = IDENTITY if point else sweep_map(size, toward, light_cell)
        blurred: bool = lit.scattering > 0.0 and lit.scattering_model == ScatteringModel.BLUR
        for light in (self._light, *self._light_blur) if blurred else (self._light,):
            if not light.allocated or (light.width, light.height, light.depth) != stored:
                light.allocate(*stored, GL_RGBA16F)
        if not blurred:
            for blur in self._light_blur:
                blur.deallocate()
        box = (1.0, 1.0 / self._aspect, self._z_extent)
        sigma, fog = self._light_extinction()
        if point:
            self._light_shader.use(density, self._light, box=box, point=True, vector=vector, sigma=sigma, fog=fog)
        else:
            self._light_sweep_shader.use(density, self._light, size, light_cell, toward=toward, sigma=sigma, fog=fog)
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)
        light: Texture3D = self._light
        if blurred:
            a, b = self._light_blur
            for axis, source, result in ((0, self._light, a), (1, a, b), (2, b, a)):
                self._light_blur_shader.use(source, self._light, result, axis, lit.scattering)
                glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)
            light = a
        self._light_shown = light
        intensity: float = lit.intensity
        return CompositeLight(light, intensity, lit.anisotropy, lit.back_scattering, point, vector, box,
                              grid=size, light_map=light_map)

    def _lit_sky(self) -> tuple[Texture3D | None, tuple[float, ...]]:
        """The light from straight above, swept at a quarter of the simulation's cells each way, soft for its
        coarseness as light from the whole sky is; with the map from a place to its sheared texel."""
        density: Texture3D | None = self._density.weight
        if density is None:
            return None, IDENTITY
        sigma, fog = self._light_extinction()
        size = (max(1, self._simulation_width // 4), max(1, self._simulation_height // 4), max(1, self._depth // 4))
        above = (0.0, -1.0, 0.0)
        sky_cell = (self._cell[0] * 4, self._cell[1] * 4, self._z_extent / size[2])
        stored = sweep_size(size, above, sky_cell)
        if not self._sky.allocated or (self._sky.width, self._sky.height, self._sky.depth) != stored:
            self._sky.allocate(*stored, GL_RGBA16F)
        self._light_sweep_shader.use(density, self._sky, size, sky_cell, toward=above, sigma=sigma, fog=fog)
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)
        return self._sky, sweep_map(size, above, sky_cell)

    def _light_place(self) -> LightPlace | None:
        """Where the light is, in the box's texture coordinates, None while it is off: a lamp at its place, a sun
        where its light, coming in toward the box's center, enters the box."""
        lit = self.settings.composite.light
        if not lit.enabled:
            return None
        vector = self._light_override or (lit.x, lit.y, lit.z)
        if lit.source == Light.POINT:
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
        for carrier in (self._density, self._fine_density, self._particle_density):
            carrier.set_walls(wall, wall)

    def _handle_deferred_actions(self) -> None:
        """Process reset and reallocation requests queued from the UI thread."""
        if self._reallocate_pending:
            self._reallocate_pending = False
            old_sim = (self._simulation_width, self._simulation_height, self._depth, self._cell)
            self._update_dimensions()
            if (self._simulation_width, self._simulation_height, self._depth, self._cell) != old_sim:
                self._allocate_simulation_fields()
                self._densities_pending = True

        if self._densities_pending:
            self._densities_pending = False
            self._allocate_densities()

        if self._densities_clear:
            self._densities_clear = False
            for carrier in self._densities:
                carrier.reset()

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
        """The saturating loss's share this step, dt / evaporation_time: a full value well above the knee is gone in
        evaporation_time; 0 off."""
        return delta_time / evaporation_time if evaporation_time > 0.0 else 0.0

    # ========== Input Methods (2D -> 3D injection) ==========

    def drag_velocity(self, texture: Texture, weight: float = 1.0, z_position: Texture | None = None) -> None:
        """The boundary: drag the velocity in the gaussian birth slab toward the body's velocity (x, y on
        the image plane, z toward the camera; widths/s) × input.velocity_gain, over input.velocity_drag, where
        the body moves (its speed above DRAG_FLOOR), never past it. The volume's w is -z, in widths/s as x
        and y (the box as deep as it stands for); +W moves fluid away from the viewer (layer 0
        is the front face).
        The slab is centered on each pixel's `z_position` (red over green); nothing is dragged where it has
        none or without one.
        """
        if not self._allocated:
            return
        input = self.settings.input
        self._add_to_faces_shader.drag(self._faces.volume, self._cells, texture, self._spread(),
                                       input.velocity_gain, drag_share(self._dt, input.velocity_drag), weight,
                                       DRAG_FLOOR, self._obstacle, self._closed, self._has_obstacles, z_position)
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

    def add_acceleration(self, texture: Texture, weight: float = 1.0, z_position: Texture | None = None) -> None:
        """The body force: add the acceleration (x, y on the image plane, z toward the camera; widths/s²)
        × input.acceleration_gain × dt into the gaussian birth slab at `z_position`; w is -z, as the
        boundary's."""
        if not self._allocated:
            return
        self._add_to_faces_shader.add_acceleration(self._faces.volume, self._cells, texture, self._spread(), weight * self.settings.input.acceleration_gain * self._dt,
                                                   self._obstacle, self._closed, self._has_obstacles, z_position)
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

    def add_density(self, amount: Texture, color: Texture, weight: float = 1.0, z_position: Texture | None = None) -> None:
        """The boundary's dye, the bridge's `amount` (red) times input.density_gain times its `color` (RGBA,
        premultiplied by its alpha, the coverage), into the carriers by input.density_blend: over the water's dye
        by input.density_drag's share, the amount capped at 1; added × dt; or the water drawn toward it, at most
        fully dyed, over input.density_drag; where the water holds more than the body, down over
        input.density_release (0 = never down)."""
        if not self._allocated:
            return
        input = self.settings.input
        blend: DensityBlend = input.density_blend
        strength: float = (self._dt if blend == DensityBlend.ADD else drag_share(self._dt, input.density_drag)) * weight
        release: float = drag_share(self._dt, input.density_release) * weight if input.density_release > 0.0 else 0.0
        for carrier in self._densities:
            carrier.add(amount, color, strength, self._dt, self._spread(), blend, input.density_gain, release,
                        z_position)

    def add_temperature(self, texture: Texture, weight: float = 1.0, z_position: Texture | None = None) -> None:
        """The boundary's warmth: the bridge's presence (0 to 1) times input.temperature_gain, its sign and size, at
        most ±1, the water in the slab dragged toward it over input.temperature_drag where the body is warmer (or
        colder) than the water; never back toward a fainter body: as the density's, held at the body, cooled by the
        fade alone."""
        if not self._allocated or not self._temperature_runs:      # nothing reads it
            return
        input = self.settings.input
        share: float = drag_share(self._dt, input.temperature_drag) * weight
        self._inject_shader.use(texture, self._temperature, self._spread(), share,
                                internal_format=GL_R16F, mode=int(InjectMode.FILL), gain=input.temperature_gain,
                                z_position=z_position)
        glMemoryBarrier(_BARRIER_IMAGE)

    def add_expansion(self, texture: Texture, weight: float = 1.0, z_position: Texture | None = None) -> None:
        """The expansion (a presence, 0 to 1) added into the expansion's slab at `z_position`, summed over this
        tick's injections; the projection makes the divergence expansion.rate (1/s at full presence) times it, while
        `expansion.enabled` is on and the rate not 0. The first injection of a tick clears the last tick's."""
        if not self._allocated or self._expansion_rate == 0.0:
            return
        if not self._expansion_fresh:
            self._expansion_vol.clear()
            self._expansion_fresh = True
        self._inject_shader.use(texture, self._expansion_vol, self._spread(), weight,
                                internal_format=GL_R16F, mode=int(InjectMode.ADD), z_position=z_position)
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)

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
        obstacle as it comes, and the running carriers' (the `density`, the `fine_density`)."""
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
                'buoyancy': Port(self._buoyancy_force_vol, TextureKind.ACCELERATION, extent, self._show_buoyancy,
                                 light=light),
                'expansion': volume(self._expansion_vol, TextureKind.SCALAR),
                'divergence': volume(self._divergence_vol, TextureKind.SCALAR),
                'pressure': volume(self._pressure, TextureKind.SCALAR),
                **({'light': Port(self._light_shown, TextureKind.IMAGE, extent, held=3, light=light)} if self._light_shown.allocated else {}),
                **{name: replace(port, light=light) if port.volume else port
                   for carrier in self._densities for name, port in carrier.ports.items()}}
