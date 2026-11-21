"""The density as particles: points in one storage buffer, a ring, emitted from the density sources, moved
through the grid's velocity, faded and drawn by `draw` and `overlap`. How and why: modules/flow/FLUID_3D.md,
"The particles".
"""

import math

from OpenGL.GL import (GL_COLOR, GL_COLOR_BUFFER_BIT, GL_LINEAR, GL_POINT_SIZE_RANGE, GL_PROGRAM_POINT_SIZE, GL_R32F,
                       GL_RGBA16F, GL_RGBA32F, GL_TEXTURE_FETCH_BARRIER_BIT, GL_SHADER_IMAGE_ACCESS_BARRIER_BIT,
                       glClear, glClearBufferfv, glClearColor, glEnable, glGetFloatv, glMemoryBarrier)

from modules.gl import Fbo, GpuTimer, LayeredFbo, StorageBuffer, BlendState, Texture, Texture3D
from ..port import Port
from ..wall_mode import WallMode
from .density_carrier import CarrierStep, DensityCarrier
from ..density_blend import DensityBlend
from .fluid_settings import Fluid3DSettings, Overlap, ParticleDraw
from .shaders import (RING_BYTES, MarchLook, CompositeLight, ParticleAdvect, ParticleComposite, ParticleEmit, ParticleRing, ParticleSprites, ParticleSurface, ParticleWeighted,
                      ParticleSurfaceShade, ParticleSurfaceSmooth)

PARTICLE_BYTES: int = 32        # position and mass/age (vec4), previous position and color (vec4)
AMOUNT_FLOOR: float = 0.01      # the amount below which a texel spawns fewer particles, each at the floor's mass
SMOOTH_SHARE: float = 0.5       # the surface's filter reaches this share of a sphere's radius each way
SMOOTH_REACH: int = 2           # and at least these pixels
SURFACE_DEPTH, SURFACE_THICKNESS, SURFACE_COLOR = 0, 1, 2      # the surface's targets (MIN, SUM, SUM)
WEIGHTED_SUM, WEIGHTED_REVEAL = 0, 1                           # the weighted look's targets
_BARRIER_FETCH_AND_IMAGE: int = int(GL_TEXTURE_FETCH_BARRIER_BIT) | int(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

_SPRITES: dict[ParticleDraw, int] = {ParticleDraw.POINTS: ParticleSprites.POINTS, ParticleDraw.STREAKS: ParticleSprites.STREAKS,
                                     ParticleDraw.BLOBS: ParticleSprites.BLOBS}


FULL_WIDTH_HALF_MAXIMUM: float = 2.0 * math.sqrt(2.0 * math.log(2.0))   # a gaussian's width at half its height, in sigmas


REFERENCE_Z_POSITION: float = 0.5     # the box's middle, a slab its walls cut nothing of: the reference mass's


def slab_sum(z_position: float, spread: float, layers: int) -> float:
    """The body's shares summed over the grid's layers, as `slab.glsl` gives them (each cell the profile's mean
    between its faces), for a slab centered at `z_position` (0 to 1): what one unit of density a step puts into a
    texel's column. `spread` the sigma, a share of the depth."""
    if spread <= 1e-6:
        return 0.0
    target: float = (0.5 + z_position * (layers - 1)) / layers
    scale: float = 1.0 / (spread * math.sqrt(2.0))
    area: float = 0.5 * spread * math.sqrt(2.0 * math.pi) * (math.erf((1.0 - target) * scale) - math.erf(-target * scale))
    return area * layers        # each cell's mean, over a cell of 1 / layers: the area over the column times the layers


class ParticleDensity(DensityCarrier):
    """The particle carrier: a ring of `particles.count` particles, emitted, moved and drawn on the GPU."""

    def __init__(self, settings: Fluid3DSettings) -> None:
        self.settings: Fluid3DSettings = settings
        self._particles: StorageBuffer = StorageBuffer()
        self._ring: StorageBuffer = StorageBuffer()
        self._count: int = 0
        self._layers: int = 1
        self._grid_width: int = 1
        self._output: tuple[int, int] = (1, 1)
        self._z_extent: float = 1.0
        self._scale: float = 1.0            # the output's pixels per emitted texel, from the last source
        self._seed: int = 0
        self._max_point_size: float = 64.0
        self._slices: LayeredFbo = LayeredFbo()
        # The surface's targets (SURFACE_DEPTH, _THICKNESS, _COLOR): the spheres' nearest depth, their thickness,
        # their color weighted; the smoothed depth
        self._surface: Fbo = Fbo(extra_targets=(GL_LINEAR, GL_LINEAR))
        self._smoothed: tuple[Texture, Texture] = (Texture(), Texture())
        # The weighted look's targets (WEIGHTED_SUM, _REVEAL): the weighted emission and opacity, and the revealage
        self._weighted: Fbo = Fbo(extra_targets=(GL_LINEAR,))
        self._emit_shader: ParticleEmit = ParticleEmit()
        self._ring_shader: ParticleRing = ParticleRing()
        self._advect_shader: ParticleAdvect = ParticleAdvect()
        self._sprites_shader: ParticleSprites = ParticleSprites()
        self._composite_shader: ParticleComposite = ParticleComposite()
        self._surface_shader: ParticleSurface = ParticleSurface()
        self._smooth_shader: ParticleSurfaceSmooth = ParticleSurfaceSmooth()
        self._shade_shader: ParticleSurfaceShade = ParticleSurfaceShade()
        self._weighted_shader: ParticleWeighted = ParticleWeighted()
        # The passes' GPU timers, inside the fluid's density and composite timers
        self._emit_timer: GpuTimer = GpuTimer("GPU particles emit")
        self._advect_timer: GpuTimer = GpuTimer("GPU particles advect")
        self._draw_timer: GpuTimer = GpuTimer("GPU particles draw")         # the sprites or the spheres into their targets
        self._march_timer: GpuTimer = GpuTimer("GPU particles march")       # the slices' march, or the surface's filter and shade

    def _shaders(self) -> tuple:
        return (self._emit_shader, self._ring_shader, self._advect_shader, self._sprites_shader, self._composite_shader,
                self._surface_shader, self._smooth_shader, self._shade_shader, self._weighted_shader)

    # ========== Lifecycle ==========

    def allocate(self, width: int, height: int, layers: int, extent: float, output_width: int, output_height: int) -> None:
        self._count = int(self.settings.particles.count)
        self._layers = layers
        self._grid_width = width
        self._output = (output_width, output_height)
        self._z_extent = extent
        self._scale = (output_width * output_height) / max(1, width * height)
        self._particles.allocate(self._count * PARTICLE_BYTES)
        self._ring.allocate(RING_BYTES)
        for shader in self._shaders():
            shader.allocate()
        glEnable(GL_PROGRAM_POINT_SIZE)
        self._max_point_size = float(glGetFloatv(GL_POINT_SIZE_RANGE)[1])

    def deallocate(self) -> None:
        self._particles.deallocate()
        self._ring.deallocate()
        self._slices.deallocate()
        self._deallocate_surface()
        self._deallocate_weighted()
        for shader in self._shaders():
            shader.deallocate()
        for timer in (self._emit_timer, self._advect_timer, self._draw_timer, self._march_timer):
            timer.deallocate()

    @property
    def allocated(self) -> bool:
        return self._particles.allocated

    def reset(self) -> None:
        self._particles.clear()
        self._ring.clear()

    # ========== The step ==========

    def _reference_mass(self) -> float:
        """The mass a texel of fully dyed water gives each of its particles: a blob of this mass has `radius`, so
        how much dye there is changes how many drops there are, not how big a drop is."""
        settings = self.settings
        spread: float = settings.input.spread / FULL_WIDTH_HALF_MAXIMUM / max(self._z_extent, 1e-6)     # the sigma, a share of the depth
        return slab_sum(REFERENCE_Z_POSITION, spread, self._layers) / max(self.settings.particles.rate, 1e-6)

    def add(self, amount: Texture, color: Texture, strength: float, dt: float, spread: float, blend: DensityBlend,
            gain: float, release: float, z_position: Texture | None = None) -> None:
        """Particles born where the dye is, as many as its amount × the rate, their mass the share of it the blend
        adds this step: over's and fill's share counted as that share of fully dyed added, as the grid's first
        step from clear water does, so a particle look sees the three alike; particles are only born, so the
        `release` has nothing to take. Their color is RGB, as the bridge's: a
        particle mixes with none."""
        if not self.allocated or not amount.allocated or not color.allocated or strength <= 0.0:
            return
        self._seed += 1
        self._scale = (self._output[0] * self._output[1]) / max(1, color.width * color.height)
        self._emit_timer.begin()
        self._emit_shader.use(amount, color, self._particles, self._ring, self._count, strength=strength,
                              blend=int(blend), gain=gain, dt=dt,
                              rate=self.settings.particles.rate, floor=AMOUNT_FLOOR,
                              layers=self._layers, spread=spread, seed=self._seed, z_position=z_position)
        self._emit_timer.end()

    def update(self, step: CarrierStep) -> None:
        if not self.allocated:
            return
        self._advect_timer.begin()
        vertices: int = 2 if self.settings.particles.draw == ParticleDraw.STREAKS else 1
        self._ring_shader.use(self._ring, self._count, vertices, ParticleAdvect.WORKGROUP_SIZE_X)
        self._advect_shader.use(self._particles, self._ring, self._count, step.velocity, aspect=step.aspect, z_extent=step.z_extent,
                                timestep=step.step, dt=step.dt, dissipation=step.dissipation,
                                evaporation=step.evaporation * self._reference_mass(),
                                evaporation_knee=step.evaporation_knee * self._reference_mass(),
                                closed_xy=self.settings.wall == WallMode.CLOSED,
                                closed_z=self.settings.wall == WallMode.CLOSED)
        self._advect_timer.end()

    # ========== Drawing ==========

    def composite(self, output: Texture, look: MarchLook, light: CompositeLight | None = None) -> None:
        """The particles into `output`; the light is not theirs yet, their look is a step of its own. The
        composite's `emission` and `absorption` are over the whole depth, so a particle look takes them over
        its `slices`."""
        if not self.allocated:
            return
        if self.settings.particles.draw == ParticleDraw.SURFACE:
            self._composite_surface(output, look)
        elif self.settings.particles.overlap == Overlap.WEIGHTED:
            self._composite_weighted(output, look)
        else:
            self._composite_sprites(output, look)

    def _composite_sprites(self, output: Texture, look: MarchLook) -> None:
        particles = self.settings.particles
        extinction, emission = look.absorption, look.emission * look.absorption     # thin density's glow over the view
        additive: bool = particles.overlap == Overlap.ADDITIVE
        slices: int = 1 if additive else int(particles.slices)
        target: LayeredFbo = self._slices
        width, height = self._output
        if not target.allocated or (target.width, target.height, target.depth) != (width, height, int(particles.slices)):
            target.deallocate()
            target.allocate(width, height, int(particles.slices), GL_RGBA16F)

        self._draw_timer.begin()
        target.begin()
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT)
        BlendState.push()
        BlendState.set_blend_mode(BlendState.BlendMode.SUM)
        self._sprites_shader.use(self._particles, self._ring, self._count, _SPRITES[particles.draw], slices=slices,
                                 scale=self._scale, ring_fade=particles.ring_fade, output_size=self._output,
                                 size=particles.size, streak=particles.streak, radius=particles.radius,
                                 reference_mass=self._reference_mass(), spread=particles.spread,
                                 fade_time=self.settings.density.fade_time, max_point_size=self._max_point_size,
                                 fog=look.fog if additive else (0.0, 0.0, 0.0))
        BlendState.pop()
        target.end()
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)
        self._draw_timer.end()

        self._march_timer.begin()
        self._composite_shader.use(target, slices, output, emission=emission / self.settings.particles.slices, absorption=extinction / self.settings.particles.slices,
                                   fog=(0.0, 0.0, 0.0) if additive else look.fog)
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)
        self._march_timer.end()

    def _composite_weighted(self, output: Texture, look: MarchLook) -> None:
        """The sprites into the weighted look's two targets in one pass, then into the output."""
        particles = self.settings.particles
        extinction, emission = look.absorption, look.emission * look.absorption     # thin density's glow over the view
        width, height = self._output
        if not self._weighted.allocated or (self._weighted.width, self._weighted.height) != (width, height):
            self._deallocate_weighted()
            self._allocate_weighted()

        self._draw_timer.begin()
        self._weighted.begin()
        glClearBufferfv(GL_COLOR, 0, (0.0, 0.0, 0.0, 0.0))
        glClearBufferfv(GL_COLOR, 1, (1.0, 1.0, 1.0, 1.0))
        BlendState.push()
        BlendState.set_blend_mode(BlendState.BlendMode.SUM)
        BlendState.set_buffer_blend_mode(1, BlendState.BlendMode.REVEALAGE)   # the revealage: × (1 - alpha) per channel
        self._sprites_shader.use(self._particles, self._ring, self._count, _SPRITES[particles.draw], slices=1,
                                 scale=self._scale, ring_fade=particles.ring_fade, output_size=self._output,
                                 size=particles.size, streak=particles.streak, radius=particles.radius,
                                 reference_mass=self._reference_mass(), spread=particles.spread,
                                 fade_time=self.settings.density.fade_time, max_point_size=self._max_point_size,
                                 fog=look.fog, weighted=True, emission=emission / self.settings.particles.slices,
                                 absorption=extinction / self.settings.particles.slices)
        BlendState.pop()
        self._weighted.end()
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)
        self._draw_timer.end()

        self._march_timer.begin()
        self._weighted_shader.use(self._weighted.targets[WEIGHTED_SUM], self._weighted.targets[WEIGHTED_REVEAL], output)
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)
        self._march_timer.end()

    def _allocate_weighted(self) -> None:
        width, height = self._output
        self._weighted.allocate(width, height, (GL_RGBA32F, GL_RGBA16F))      # the weight reaches 3000: past a half float's sum

    def _deallocate_weighted(self) -> None:
        self._weighted.deallocate()

    def _allocate_surface(self) -> None:
        width, height = self._output
        self._surface.allocate(width, height, (GL_R32F, GL_R32F, GL_RGBA16F))
        for smoothed in self._smoothed:
            smoothed.allocate(width, height, GL_R32F)

    def _deallocate_surface(self) -> None:
        self._surface.deallocate()
        for texture in self._smoothed:
            texture.deallocate()

    def _composite_surface(self, output: Texture, look: MarchLook) -> None:
        particles = self.settings.particles
        extinction, emission = look.absorption, look.emission * look.absorption     # thin density's glow over the view
        width, height = self._output
        if not self._surface.allocated or (self._surface.width, self._surface.height) != (width, height):
            self._deallocate_surface()
            self._allocate_surface()

        self._draw_timer.begin()
        self._surface.begin()
        glClearBufferfv(GL_COLOR, 0, (1.0, 1.0, 1.0, 1.0))
        glClearBufferfv(GL_COLOR, 1, (0.0, 0.0, 0.0, 0.0))
        glClearBufferfv(GL_COLOR, 2, (0.0, 0.0, 0.0, 0.0))
        BlendState.push()
        BlendState.set_blend_mode(BlendState.BlendMode.SUM)
        BlendState.set_buffer_blend_equation(0, BlendState.BlendEquation.MIN)
        self._surface_shader.use(self._particles, self._ring, self._count, scale=self._scale, ring_fade=particles.ring_fade,
                                 z_per_pixel=1.0 / (width * self._z_extent),
                                 radius=particles.radius, reference_mass=self._reference_mass(), spread=particles.spread,
                                 fade_time=self.settings.density.fade_time, max_point_size=self._max_point_size)
        BlendState.pop()
        self._surface.end()
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)
        self._draw_timer.end()

        self._march_timer.begin()
        # The filter reaches half a sphere's radius each way, a sphere's depth its range: neighbors melt, a
        # drop's edge against the empty water stays
        depth: Texture = self._surface.targets[SURFACE_DEPTH]
        reach: int = max(SMOOTH_REACH, round(SMOOTH_SHARE * particles.radius))
        depth_range: float = 2.0 * particles.radius / (width * self._z_extent)
        for k in range(2 * int(particles.smooth)):
            target: Texture = self._smoothed[k % 2]
            self._smooth_shader.use(depth, target, k % 2 == 0, reach, 0.5 * reach, depth_range)
            glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)
            depth = target
        self._shade_shader.use(depth, self._surface.targets[SURFACE_THICKNESS], self._surface.targets[SURFACE_COLOR], output,
                               depth_per_pixel=width * self._z_extent, emission=emission / self.settings.particles.slices,
                               absorption=extinction / self.settings.particles.slices, fog=look.fog)
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)
        self._march_timer.end()

    # ========== Outputs ==========

    @property
    def weight(self) -> Texture3D | None:
        return None

    @property
    def ports(self) -> dict[str, Port]:
        return {}
