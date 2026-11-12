"""The 3D fluid's settings: the box, then one group per concern (time, velocity, density, temperature,
pressure, composite, fine_density, particles), with the concern's name as the group's, so the field names carry no prefix."""

from enum import IntEnum

from modules.settings import Field, BaseSettings, Group, Widget
from modules.utils import Color

from ..wall_mode import WallMode
from ..time_step import TimeStepMode


class Advection(IntEnum):
    """How the velocity, the density and the temperature are carried each step."""
    SEMI_LAGRANGIAN = 0     # trace back and read, trilinear: stable, blurs a little every step
    MACCORMACK = 1          # the same, corrected by half its error traced forward, clamped (Selle 2008)


class Limiter(IntEnum):
    """What MacCormack does where its correction leaves the eight texels the plain step read between (Selle 2008)."""
    CLAMP = 0       # the corrected value cut to those bounds: the edge kept steep, its top flat
    REVERT = 1      # the plain step's value there: what a step really made, softer at the sharpest features


class PressureSolver(IntEnum):
    """How the pressure's Poisson equation is solved each step, warm from the last step's pressure."""
    JACOBI = 0              # every cell from its neighbors' last values; five iterations a dispatch in shared memory
    SOR = 1                 # red-black Gauss–Seidel, over-relaxed: each color from the other's newest values
    MULTIGRID = 2           # V-cycles over the grid halved to a few cells, red-black Gauss–Seidel smoothing each level


class Light(IntEnum):
    """The light on the density."""
    DIRECTIONAL = 0         # a sun: one direction everywhere, parallel shadows
    POINT = 1               # a lamp at one place: shadows fanning out from it


class LightGrid(IntEnum):
    """The light grid's cells along each axis, as a share of the simulation's."""
    QUARTER = 4
    HALF = 2
    FULL = 1


class ScatteringModel(IntEnum):
    """How light bounced inside the density is made."""
    BLUR = 0                # the light grid blurred and mixed in: soft, but it spreads light into empty water
    OCTAVES = 1             # octaves of the light with less extinction summed (Wrenninge): thick parts glow within


class DensitySample(IntEnum):
    """Which velocity moves a fine density layer that spans several of the simulation's layers."""
    MIDDLE = 0              # the velocity at the layer's middle
    AVERAGE = 1             # the velocity averaged over the grid's layers the layer spans


class ParticleDraw(IntEnum):
    """How the particles are drawn."""
    POINTS = 0              # one pixel a particle
    STREAKS = 1             # a line from the previous to the current position
    BLOBS = 2               # a gaussian a particle, the sum a soft melt
    SURFACE = 3             # spheres melted into one surface, shaded (screen-space fluid rendering)


class Overlap(IntEnum):
    """How the particles cover each other, for points, streaks and blobs."""
    OCCLUDED = 0            # depth slices marched front to back: near density covers far density
    ADDITIVE = 1            # one sum, the depth as fog per particle
    WEIGHTED = 2            # weighted blended order-independent transparency: one pass, near density over far by a depth weight (McGuire, Bavoil 2013)


class ParticleSettings(BaseSettings):
    """The particles: the ring, the emission, the look; drawn over the volume."""
    enabled: Field[bool]            = Field(False,                                                      description="Particles carried by the water, drawn over the volume")
    count: Field[int]              = Field(2_000_000, min=10_000, max=8_000_000, step=10_000,     description="Particles in the ring, the most alive at once; 32 bytes each")
    rate: Field[float]              = Field(20.0,   min=0.01,   max=50.0,                               description="Particles per texel per second where density is (fewer: heavier, bigger blobs)")
    ring_fade: Field[float]         = Field(0.1,    min=0.01,   max=1.0,                                description="The ring's last share the oldest fade over before they are overwritten")
    draw: Field[ParticleDraw]       = Field(ParticleDraw.POINTS,                                        description="Points, streaks, blobs or a surface")
    overlap: Field[Overlap]         = Field(Overlap.OCCLUDED,                                           description="Occluded: near density covers far density. Additive: one sum. Weighted: near over far by a weight, one pass")
    slices: Field[int]              = Field(16,     min=2,      max=32,                                 description="Depth slices of the occluded look, at the output's size")
    size: Field[float]              = Field(1.0,    min=1.0,    max=8.0,                                description="A point's size in pixels")
    streak: Field[float]            = Field(1.0,    min=0.0,    max=4.0,                                description="A streak's length as a share of its step")
    radius: Field[float]            = Field(24.0,   min=1.0,    max=48.0,                               description="A blob's or a sphere's radius in output pixels at the reference mass")
    spread: Field[float]            = Field(2.0,    min=0.0,    max=10.0,                               description="A blob's widening with age (pixels² per second), at most twice its radius")
    smooth: Field[int]              = Field(3,      min=0,      max=8,                                  description="The surface's smoothing passes")


class TimeSettings(BaseSettings):
    """The fluid's time: how a step's length is taken, and the fluid's time against real time."""
    time_step: Field[TimeStepMode]  = Field(TimeStepMode.MEASURED,                                  description="Step by the measured render rate, or by a fixed rate")
    fixed_fps: Field[float]         = Field(60.0,   min=1.0,    max=240.0,                          description="Steps per second in fixed mode (fps)")
    fps_smoothing: Field[float]     = Field(1.0,    min=0.0,    max=10.0,                           description="Smoothing of the measured rate in measured mode (s); 0 follows each frame")
    time_scale: Field[float]        = Field(1.0,    min=0.05,   max=2.0,    step=0.05,              description="The fluid's time as a share of real time; 0.5 = everything at half speed, 1 = real time")


class BoundarySettings(BaseSettings):
    """The boundary, the body in the water: what the water is dragged toward, each field a line, its target the
    body's value times a gain (capped at fully dyed and ±1 warm) and the time it takes to follow."""
    velocity_drag: Field[float]     = Field(50.0,   min=0.0,    max=1000.0, step=1.0, color="cyan", newline=True, description="Time the water takes to close the gap to the body's velocity (ms); 0 = at once")
    velocity_gain: Field[float]     = Field(1.0,    min=0.0,    max=10.0,   color="teal",           description="The body's velocity times this is what the water is dragged toward; 1 = as fast as the body")
    density_drag: Field[float]      = Field(200.0,  min=0.0,    max=1000.0, step=1.0, color="cyan", newline=True, description="Time the water takes to take on the body's dye (ms); 0 = at once")
    density_gain: Field[float]      = Field(1.0,    min=0.0,    max=10.0,   color="teal",           description="The body's dye times this, at most fully dyed, is the target; 1 = fully dyed at 1 width/s")
    temperature_drag: Field[float]  = Field(200.0,  min=0.0,    max=1000.0, step=1.0, color="cyan", newline=True, description="Time the water takes to take on the body's warmth (ms); 0 = at once")
    temperature_gain: Field[float]  = Field(1.0,    min=0.0,    max=10.0,   color="teal",           description="The body's warmth times this, at most ±1, is the target; 1 = fully warm at 1 width/s")


class BodyForceSettings(BaseSettings):
    """The body force, the Navier–Stokes force term: the acceleration added × dt, a push's dye and warmth at a rate."""
    acceleration_gain: Field[float] = Field(1.0,    min=0.0,    max=10.0,   color="teal",           description="The body force's acceleration times this is added × dt; 1 = its Δv added once")
    density_rate: Field[float]      = Field(1.0,    min=0.0,    max=20.0,   color="teal",           description="Fully dyed water a full push adds per second, never past fully dyed")
    temperature_rate: Field[float]  = Field(1.0,    min=0.0,    max=20.0,   color="teal",           description="Full warmth a full push adds per second, never past ±1")


class InputSettings(BaseSettings):
    """The bridge's input: where the body sits in the box's depth and how thick it is, and each system's gains."""
    layer: Field[float]             = Field(0.5,    min=0.0,    max=1.0,                            description="Normalized depth of the injection's center, where the body sits")
    spread: Field[float]            = Field(0.2,    min=0.0,    max=1.0,    step=0.001,             description="The body's thickness in z, widths, where it is at least half there; 0 = none")
    boundary: Group[BoundarySettings]       = Group(BoundarySettings)
    body_force: Group[BodyForceSettings]    = Group(BodyForceSettings)


class VelocitySettings(BaseSettings):
    """The velocity: how it carries itself, fades, swirls and thickens."""
    advection: Field[Advection]     = Field(Advection.SEMI_LAGRANGIAN,                  description="How the velocity carries itself: MacCormack keeps swirls, at about twice the reads")
    limiter: Field[Limiter]         = Field(Limiter.CLAMP,                              description="MacCormack where its correction overshoots: clamp it to the step's bounds, or revert to the plain step")
    advection_scale: Field[float]   = Field(1.0,    min=0.0,    max=1.0,    color="blue-grey",      description="How far the velocity carries itself per step; 1 = with the water, the physics")
    fade_time: Field[float]         = Field(60.0,   min=0.0,    max=300.0,  step=1.0, color="deep-blue", description="Seconds until the velocity fades to 1%, a share at a time; 0 = off")
    evaporation_time: Field[float]  = Field(0.0,    min=0.0,    max=300.0,  step=1.0, color="grey", description="Seconds 1 width/s takes to brake to rest, a fixed amount at a time; 0 = off")
    viscosity: Field[float]         = Field(0.0,    min=0.0,    max=0.01,   step=0.0001, color="indigo", description="Kinematic viscosity (widths² per second); 0 = water at this scale")
    viscosity_iter: Field[int]      = Field(20,     min=1,      max=60,     color="orange",         description="Viscosity solver iterations at 60 fps; fewer at a higher rate, never more")


class VorticitySettings(BaseSettings):
    """The vorticity confinements (Fedkiw 2001), each keeping swirls of one size alive: the strengths on a row,
    each radius under its strength."""
    strength_1: Field[float]  = Field(0.0,    min=0.0,    max=100.0,  color="purple",         description="Confinement 1's epsilon, the force epsilon × cell × (N × curl); 0 = off")
    strength_2: Field[float]  = Field(0.0,    min=0.0,    max=100.0,  color="purple",         description="Confinement 2's epsilon; 0 = off")
    strength_3: Field[float]  = Field(0.0,    min=0.0,    max=100.0,  color="purple",         description="Confinement 3's epsilon; 0 = off")
    radius_1: Field[float]    = Field(0.004,  min=0.001,  max=0.1,    newline=True, widget=Widget.log_slider, color="deep-purple", description="The size of the swirls confinement 1 keeps, widths; at least one cell")
    radius_2: Field[float]    = Field(0.004,  min=0.001,  max=0.1,    widget=Widget.log_slider, color="deep-purple", description="The size of the swirls confinement 2 keeps, widths")
    radius_3: Field[float]    = Field(0.004,  min=0.001,  max=0.1,    widget=Widget.log_slider, color="deep-purple", description="The size of the swirls confinement 3 keeps, widths")


class DensitySettings(BaseSettings):
    """The density: its size, how it is carried, fades and evaporates."""
    width: Field[int]               = Field(448,    min=32,     max=4096,   step=8,                 description="The density's width; the dye's is the bridge's width")
    height: Field[int]              = Field(800,    min=32,     max=4096,   step=8,                 description="The density's height; the dye's is the bridge's height")
    depth: Field[int]               = Field(24,     min=4,      max=256,    step=4,                 description="The density's depth in z")
    advection: Field[Advection]     = Field(Advection.SEMI_LAGRANGIAN,  newline=True,   description="How the density and the fine density are carried: MacCormack keeps their edges sharp")
    limiter: Field[Limiter]         = Field(Limiter.CLAMP,                              description="MacCormack where its correction overshoots: clamp it to the step's bounds, or revert to the plain step")
    advection_scale: Field[float]   = Field(1.0,    min=0.0,    max=5.0,    color="blue-grey",      description="How far the density is carried per step; 1 = with the water, above it runs ahead")
    fade_time: Field[float]         = Field(0.0,    min=0.0,    max=300.0,  step=1.0, color="deep-blue", description="Seconds until the density fades to 1%, a share at a time; 0 = off")
    evaporation_time: Field[float]  = Field(0.0,    min=0.0,    max=300.0,  step=1.0, color="grey", description="Seconds fully dyed water takes to clear, faint first; 0 = off")


class FineDensitySettings(BaseSettings):
    """The fine density: its size, how its layers move, how the bridge's density is enlarged, the swirls."""
    enabled: Field[bool]            = Field(False,                                                  description="The volume drawn from the fine density, its color the density's, instead of the density")
    width: Field[int]               = Field(2160,   min=32,     max=4096,   step=8,                 description="The fine density's width; the output's is 2160")
    height: Field[int]              = Field(3840,   min=32,     max=4096,   step=8,                 description="The fine density's height; the output's is 3840")
    depth: Field[int]               = Field(16,     min=4,      max=256,    step=4,                 description="The fine density's depth in z")
    sample: Field[DensitySample]    = Field(DensitySample.AVERAGE,                                  description="The velocity at a layer's middle, or averaged over the simulation's layers it spans")
    cubic: Field[bool]              = Field(True,                                                   description="The bridge's density enlarged by a cubic B-spline, without its texels' steps; off: bilinear")
    turbulence: Field[float]        = Field(0.0,    min=0.0,    max=50.0,   step=0.5,              description="Swirls below the simulation's cells where the water churns, as its curl × cell; 0 = off")


class TemperatureSettings(BaseSettings):
    """The temperature: how it is carried, fades, and lifts or sinks the water."""
    advection: Field[Advection]     = Field(Advection.SEMI_LAGRANGIAN,                  description="How the warmth is carried; only its buoyancy is seen")
    limiter: Field[Limiter]         = Field(Limiter.CLAMP,                              description="MacCormack where its correction overshoots: clamp it to the step's bounds, or revert to the plain step")
    advection_scale: Field[float]   = Field(1.0,    min=0.0,    max=5.0,    color="blue-grey",      description="How far the temperature is carried per step; 1 = with the water, above it runs ahead")
    fade_time: Field[float]         = Field(3.0,    min=0.0,    max=300.0,  step=1.0, color="deep-blue", description="Seconds until the warmth fades to 1%, a share at a time; 0 = off")
    evaporation_time: Field[float]  = Field(0.0,    min=0.0,    max=300.0,  step=1.0, color="grey", description="Seconds full warmth (±1) takes to reach 0, a fixed amount at a time; 0 = off")
    buoyancy: Field[float]          = Field(0.0,    min=0.0,    max=10.0,   color="deep-orange",    description="Lift per unit of temperature above ambient (widths/s²) against gravity; 0 = none")
    sink: Field[float]              = Field(0.0,    min=0.0,    max=20.0,   color="red",            description="Weight per unit of density (widths/s²) along gravity; 0 = as heavy as the water")
    gravity_x: Field[float]         = Field(0.0,    min=-1.0,   max=1.0,    color="deep-orange",    description="Gravity direction X")
    gravity_y: Field[float]         = Field(1.0,    min=-1.0,   max=1.0,    color="deep-orange",    description="Gravity direction Y; 1 = down the screen")
    gravity_z: Field[float]         = Field(0.0,    min=-1.0,   max=1.0,    color="deep-orange",    description="Gravity direction Z; 1 = away from viewer")
    ambient: Field[float]           = Field(0.0,    min=0.0,    max=1.0,    color="brown",          description="Reference temperature (buoyancy = 0 at this temp)")


class PressureSettings(BaseSettings):
    """Pressure field parameters."""
    solver: Field[PressureSolver]   = Field(PressureSolver.JACOBI,                                  description="Jacobi, red-black SOR (Gauss–Seidel over-relaxed) or multigrid V-cycles")
    iterations: Field[int]          = Field(40,     min=1,      max=60,     color="orange",         description="Pressure solver iterations per step; for SOR its red-black sweeps")
    relaxation: Field[float]        = Field(1.8,    min=1.0,    max=1.85,   color="orange",         description="SOR's over-relaxation factor omega; 1 = Gauss–Seidel; 1.95 diverged")
    cycles: Field[int]              = Field(1,      min=1,      max=8,      color="orange",         description="Multigrid V-cycles per step")


class LightSettings(BaseSettings):
    """The light on the density and in the water: a sun or a lamp, its shadows, the rays, the caustics and the
    bounced light; off, the composite is the plain march, every light pass skipped and its memory released."""
    enabled: Field[bool]            = Field(False,                                              description="The light on: the dye lit, its shadows, the rays and the caustics")
    source: Field[Light]            = Field(Light.DIRECTIONAL,                                  description="A sun from one direction, or a lamp at one place")
    x: Field[float]                 = Field(0.0,  min=-1.0, max=1.0,    color="amber",          description="Toward the sun, or the lamp's place (±1 the box's sides); x right")
    y: Field[float]                 = Field(-1.0, min=-1.0, max=1.0,    color="amber",          description="y down the screen; -1 = from above")
    z: Field[float]                 = Field(-0.3, min=-1.0, max=1.0,    color="amber",          description="z away from the viewer; -1 = from the front")
    grid: Field[LightGrid]          = Field(LightGrid.FULL,                                     description="The light's grid: a quarter, half or all of the simulation's cells each way; finer, sharper shadows")
    intensity: Field[float]         = Field(1.0,  min=0.0,  max=50.0,   step=0.1, color="amber", description="How bright the light shows fully dyed water filling the view, lit straight on: 1 = its color")
    fog_scattering: Field[float]    = Field(0.0,  min=0.0,  max=1.0,    color="light-blue",     description="Share of the light the fog takes out that it sends on: the water glows, shafts in the shadows; 0 = off")
    caustics: Field[float]          = Field(0.0,  min=0.0,  max=1.0,    color="amber",          description="Contrast of the caustic net, the light under a wavy surface, carried down as shafts; 0 = off")
    caustic_size: Field[float]      = Field(0.08, min=0.01, max=1.0,    widget=Widget.log_slider, color="amber", description="The caustic net's cell, widths")
    caustic_depth: Field[float]     = Field(0.5,  min=0.05, max=4.0,    widget=Widget.log_slider, color="amber", description="Widths down the light's way over which the caustic net fades to 37%")
    caustic_speed: Field[float]     = Field(0.15, min=0.0,  max=2.0,    color="amber",          description="How fast the caustic net shifts: its points' turns per second")
    scattering_model: Field[ScatteringModel] = Field(ScatteringModel.BLUR,                   description="How the bounced light is made: the light grid blurred, or octaves of less extinction")
    scattering: Field[float]        = Field(0.0,  min=0.0,  max=1.0,    color="light-blue",     description="Light bounced inside the density: soft shadows, thick parts glowing within; 0 = off")
    powder: Field[float]            = Field(0.0,  min=0.0,  max=1.0,    color="blue-grey",      description="Edges facing the light darker, thin dye glowing less than thick (Beer–powder); 0 = off")
    sky_fill: Field[float]          = Field(0.0,  min=0.0,  max=1.0,    color="light-blue",     description="Light from the sky straight above, a share of the light's intensity, so no shadow is black; 0 = off")
    anisotropy: Field[float]        = Field(0.0,  min=0.0,  max=0.9,    color="yellow",         description="Henyey–Greenstein g: 0 scatters evenly, higher forward, a silver lining against the light")
    back_scattering: Field[float]   = Field(0.0,  min=0.0,  max=1.0,    color="yellow",         description="Share scattered back toward the light by a second lobe as strong, lit from the front; 0 = off")


class CompositeSettings(BaseSettings):
    """The composite: how the density volume is marched front to back into the output, and the light on it."""
    emission: Field[float]          = Field(1.0,  min=0.0,  max=20.0,   step=0.01, color="amber", description="Glow of thick density: 1 = its own color, 0 = none")
    opacity: Field[float]           = Field(0.1,  min=0.001, max=10.0,  widget=Widget.log_slider, color="deep-orange", description="Widths of fully dyed water that hide half of what is behind it")
    fog: Field[float]               = Field(1.0,  min=0.01, max=1000.0, widget=Widget.log_slider, color="blue-grey", description="Widths of water that dim light to half")
    tint: Field[Color]              = Field(Color(1.0, 1.0, 1.0), widget=Widget.color,         description="The water's color, the hue light turns in it; white = none")
    light: Group[LightSettings]     = Group(LightSettings)


class Fluid3DSettings(BaseSettings):
    """The 3D fluid's configuration: the box, then one group per concern.

    Fade_time model:
        Exponential frame-rate-independent decay: multiplier = 0.01^(dt/fade_time), 0 = off.
        fade_time=3.0 means the field retains ~1% after 3 seconds. The evaporation takes a field's full
        value (fully dyed, ±1 warm, 1 width/s) to 0 in evaporation_time, dt / evaporation_time a step.
    """

    # ---- The guard, how the fields are carried and walled, then the box (multiples of 32) ----
    guard: Field[bool]              = Field(True,                                       description="Set non-finite values to 0, count and log them; off once the fluid is stable")
    wall: Field[WallMode]           = Field(WallMode.CLOSED,                            newline=True, description="Every side of every field. Closed: solid, the water slides along. Open: it leaves")
    slip_time: Field[float]         = Field(0.0,    min=0.0,    max=60.0,   step=0.1,   description="Seconds the water along a closed wall takes to slow to 1%; 0 = free-slip")
    width: Field[int]               = Field(256,    min=32,     max=4096,   step=32,    newline=True, description="Simulation grid width")
    height: Field[int]              = Field(448,    min=32,     max=4096,   step=32,    description="Simulation grid height")
    depth: Field[int]               = Field(128,    min=4,      max=256,    step=4,     description="Simulation grid depth in z, its cells")
    depth_ratio: Field[float]       = Field(1.0,    min=0.05,   max=4.0,    step=0.01,  description="The box's depth over its width, 1 = as deep as it is wide; the z cells stretch to it")
    z_stretch: Field[float]         = Field(1.0,    access=Field.READ,                  description="How many times longer a z cell is than an xy cell")

    # ---- The groups ----
    time: Group[TimeSettings]               = Group(TimeSettings)
    input: Group[InputSettings]             = Group(InputSettings)
    # the fluid's passes in their order: the velocity, its confinement, the temperature's buoyancy, the projection, the density
    velocity: Group[VelocitySettings]       = Group(VelocitySettings)
    vorticity: Group[VorticitySettings]     = Group(VorticitySettings)
    temperature: Group[TemperatureSettings] = Group(TemperatureSettings)
    pressure: Group[PressureSettings]       = Group(PressureSettings)
    density: Group[DensitySettings]         = Group(DensitySettings)
    fine_density: Group[FineDensitySettings] = Group(FineDensitySettings)
    particles: Group[ParticleSettings]      = Group(ParticleSettings)
    composite: Group[CompositeSettings]     = Group(CompositeSettings)
