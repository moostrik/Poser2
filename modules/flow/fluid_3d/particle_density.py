"""The density as particles: millions of points in one storage buffer, moved through the grid's velocity.

Everything stays on the GPU: the CPU dispatches and draws a fixed count and reads no particle back.
    emission     per texel of a density source, `rate` × dt particles where ink is (fewer below a floor),
                 each the texel's amount × gain × dt × the slab's sum over the grid's layers over the
                 expected count as its mass: a texel's mass is the grid's in expectation, a strong drop
                 heavy particles; into a ring by an atomic head, the oldest overwritten
    the ring     a particle's place, (head − 1 − slot) mod count over count, fades it over the ring's last
                 `ring_fade` before the ring overwrites it
    advection    RK2 through the velocity, the grid advection's uv metric; the mass faded by
                 `density.fade_time` and evaporated; a closed wall stops it, an open one takes it
    drawing      from the buffer by gl_VertexID, no copy: points, streaks or blobs into depth slices at the
                 output's size (a layered draw), marched front to back as the grid is (`OCCLUDED`), or into
                 one slice with the fog per particle (`ADDITIVE`); or spheres melted into one shaded
                 surface (`SURFACE`, screen-space fluid rendering)
A particle's value is its mass × the output's pixels per emitted texel, so its texel's particles sum to
the grid's density there, whatever the look.
"""

import math

from OpenGL.GL import (GL_COLOR, GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_BUFFER_BIT,
                       GL_FRAMEBUFFER, GL_FUNC_ADD, GL_MIN, GL_POINT_SIZE_RANGE, GL_PROGRAM_POINT_SIZE, GL_R32F,
                       GL_ONE_MINUS_SRC_COLOR, GL_RGBA16F, GL_RGBA32F, GL_TEXTURE_2D, GL_ZERO, GL_TEXTURE_FETCH_BARRIER_BIT, GL_SHADER_IMAGE_ACCESS_BARRIER_BIT,
                       glBindFramebuffer, glBlendEquationi, glBlendFunci, glClear, glClearBufferfv, glClearColor,
                       glDeleteFramebuffers, glDrawBuffers, glEnable, glFramebufferTexture2D, glGenFramebuffers,
                       glGetFloatv, glMemoryBarrier, glViewport)

from modules.gl import GpuTimer, LayeredFbo, StorageBuffer, Style, Texture, Texture3D
from ..port import Port
from ..wall_mode import WallMode
from .density_carrier import CarrierStep, DensityCarrier
from .fluid_settings import Fluid3DSettings, Overlap, ParticleDraw
from .shaders import (RING_BYTES, MarchLook, CompositeLight, InjectMode, ParticleAdvect, ParticleComposite, ParticleEmit, ParticleRing, ParticleSprites, ParticleSurface, ParticleWeighted,
                      ParticleSurfaceShade, ParticleSurfaceSmooth)

PARTICLE_BYTES: int = 32        # position and mass/age (vec4), previous position and color (vec4)
AMOUNT_FLOOR: float = 0.01      # the amount below which a texel spawns fewer particles, each at the floor's mass
SMOOTH_SHARE: float = 0.5       # the surface's filter reaches this share of a sphere's radius each way
SMOOTH_REACH: int = 2           # and at least these pixels
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
        # The surface's targets: the spheres' nearest depth, their thickness, their color weighted; the smoothed depth
        self._surface_fbo: int = 0
        self._surface_depth: Texture = Texture()
        self._surface_thickness: Texture = Texture()
        self._surface_color: Texture = Texture()
        self._smoothed: tuple[Texture, Texture] = (Texture(), Texture())
        # The weighted look's targets: the weighted emission and opacity, and the revealage
        self._weighted_fbo: int = 0
        self._weighted_sum: Texture = Texture()
        self._weighted_reveal: Texture = Texture()
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

    def reload(self) -> None:
        for shader in self._shaders():
            shader.reload()

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

    def add(self, texture: Texture, gain: float, dt: float, spread: float, mode: InjectMode = InjectMode.RATE,
            target_gain: float = 1.0, z_position: Texture | None = None) -> None:
        """Particles born where the dye is, as many as its presence × the rate; a FILL's share counted as that
        share of fully dyed added in the step, so a particle look sees both systems alike."""
        if mode == InjectMode.FILL:
            gain = gain * target_gain / max(dt, 1e-6)
        if not self.allocated or not texture.allocated or gain <= 0.0:
            return
        self._seed += 1
        self._scale = (self._output[0] * self._output[1]) / max(1, texture.width * texture.height)
        self._emit_timer.begin()
        self._emit_shader.use(texture, self._particles, self._ring, self._count, gain=gain, dt=dt,
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
        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.SUM)
        self._sprites_shader.use(self._particles, self._ring, self._count, _SPRITES[particles.draw], slices=slices,
                                 scale=self._scale, ring_fade=particles.ring_fade, output_size=self._output,
                                 size=particles.size, streak=particles.streak, radius=particles.radius,
                                 reference_mass=self._reference_mass(), spread=particles.spread,
                                 fade_time=self.settings.density.fade_time, max_point_size=self._max_point_size,
                                 fog=look.fog if additive else (0.0, 0.0, 0.0))
        Style.pop_style()
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
        if not self._weighted_fbo or (self._weighted_sum.width, self._weighted_sum.height) != (width, height):
            self._deallocate_weighted()
            self._allocate_weighted()

        self._draw_timer.begin()
        glBindFramebuffer(GL_FRAMEBUFFER, self._weighted_fbo)
        glViewport(0, 0, width, height)
        glClearBufferfv(GL_COLOR, 0, (0.0, 0.0, 0.0, 0.0))
        glClearBufferfv(GL_COLOR, 1, (1.0, 1.0, 1.0, 1.0))
        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.SUM)
        glBlendFunci(1, GL_ZERO, GL_ONE_MINUS_SRC_COLOR)        # the revealage: × (1 - alpha) per channel
        self._sprites_shader.use(self._particles, self._ring, self._count, _SPRITES[particles.draw], slices=1,
                                 scale=self._scale, ring_fade=particles.ring_fade, output_size=self._output,
                                 size=particles.size, streak=particles.streak, radius=particles.radius,
                                 reference_mass=self._reference_mass(), spread=particles.spread,
                                 fade_time=self.settings.density.fade_time, max_point_size=self._max_point_size,
                                 fog=look.fog, weighted=True, emission=emission / self.settings.particles.slices,
                                 absorption=extinction / self.settings.particles.slices)
        Style.pop_style()
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)
        self._draw_timer.end()

        self._march_timer.begin()
        self._weighted_shader.use(self._weighted_sum, self._weighted_reveal, output)
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)
        self._march_timer.end()

    def _allocate_weighted(self) -> None:
        width, height = self._output
        self._weighted_sum.allocate(width, height, GL_RGBA32F)      # the weight reaches 3000: past a half float's sum
        self._weighted_reveal.allocate(width, height, GL_RGBA16F)
        self._weighted_fbo = int(glGenFramebuffers(1))
        glBindFramebuffer(GL_FRAMEBUFFER, self._weighted_fbo)
        for attachment, texture in ((GL_COLOR_ATTACHMENT0, self._weighted_sum), (GL_COLOR_ATTACHMENT1, self._weighted_reveal)):
            glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_2D, texture.tex_id, 0)
        glDrawBuffers(2, [GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1])
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def _deallocate_weighted(self) -> None:
        if self._weighted_fbo:
            glDeleteFramebuffers(1, [self._weighted_fbo])
            self._weighted_fbo = 0
        for texture in (self._weighted_sum, self._weighted_reveal):
            texture.deallocate()

    def _allocate_surface(self) -> None:
        width, height = self._output
        self._surface_depth.allocate(width, height, GL_R32F)
        self._surface_thickness.allocate(width, height, GL_R32F)
        self._surface_color.allocate(width, height, GL_RGBA16F)
        for smoothed in self._smoothed:
            smoothed.allocate(width, height, GL_R32F)
        self._surface_fbo = int(glGenFramebuffers(1))
        glBindFramebuffer(GL_FRAMEBUFFER, self._surface_fbo)
        for attachment, texture in ((GL_COLOR_ATTACHMENT0, self._surface_depth), (GL_COLOR_ATTACHMENT1, self._surface_thickness),
                                    (GL_COLOR_ATTACHMENT2, self._surface_color)):
            glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_2D, texture.tex_id, 0)
        glDrawBuffers(2, [GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1])
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def _deallocate_surface(self) -> None:
        if self._surface_fbo:
            glDeleteFramebuffers(1, [self._surface_fbo])
            self._surface_fbo = 0
        for texture in (self._surface_depth, self._surface_thickness, self._surface_color, *self._smoothed):
            texture.deallocate()

    def _composite_surface(self, output: Texture, look: MarchLook) -> None:
        particles = self.settings.particles
        extinction, emission = look.absorption, look.emission * look.absorption     # thin density's glow over the view
        width, height = self._output
        if not self._surface_fbo or (self._surface_depth.width, self._surface_depth.height) != (width, height):
            self._deallocate_surface()
            self._allocate_surface()

        self._draw_timer.begin()
        glBindFramebuffer(GL_FRAMEBUFFER, self._surface_fbo)
        glViewport(0, 0, width, height)
        glClearBufferfv(GL_COLOR, 0, (1.0, 1.0, 1.0, 1.0))
        glClearBufferfv(GL_COLOR, 1, (0.0, 0.0, 0.0, 0.0))
        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.SUM)
        glBlendEquationi(0, GL_MIN)
        self._surface_shader.use(self._particles, self._ring, self._count, scale=self._scale, ring_fade=particles.ring_fade,
                                 output_size=self._output, z_per_pixel=1.0 / (width * self._z_extent),
                                 radius=particles.radius, reference_mass=self._reference_mass(), spread=particles.spread,
                                 fade_time=self.settings.density.fade_time, max_point_size=self._max_point_size)
        glBlendEquationi(0, GL_FUNC_ADD)
        Style.pop_style()
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)
        self._draw_timer.end()

        self._march_timer.begin()
        # The filter reaches half a sphere's radius each way, a sphere's depth its range: neighbors melt, a
        # drop's edge against the empty water stays
        depth: Texture = self._surface_depth
        reach: int = max(SMOOTH_REACH, round(SMOOTH_SHARE * particles.radius))
        depth_range: float = 2.0 * particles.radius / (width * self._z_extent)
        for k in range(2 * int(particles.smooth)):
            target: Texture = self._smoothed[k % 2]
            self._smooth_shader.use(depth, target, k % 2 == 0, reach, 0.5 * reach, depth_range)
            glMemoryBarrier(_BARRIER_FETCH_AND_IMAGE)
            depth = target
        self._shade_shader.use(depth, self._surface_thickness, self._surface_color, output,
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
