"""The 3D fluid's settings: one group per field (velocity, density, temperature, pressure) and
the volume, with the field's name as the group's, so the field names carry no prefix."""

from enum import IntEnum

from modules.settings import Field, BaseSettings, Group

from ..wall_mode import WallMode
from ..time_step import TimeStepMode


class Advection(IntEnum):
    """How the velocity, the density and the temperature are carried each step."""
    SEMI_LAGRANGIAN = 0     # trace back and read, trilinear: stable, blurs a little every step
    MACCORMACK = 1          # the same, corrected by half its error traced forward, clamped (Selle 2008)


class PressureSolver(IntEnum):
    """How the pressure's Poisson equation is solved each step, warm from the last step's pressure."""
    JACOBI = 0              # every cell from its neighbors' last values; five iterations a dispatch in shared memory
    SOR = 1                 # red-black Gauss–Seidel, over-relaxed: each color from the other's newest values
    MULTIGRID = 2           # V-cycles over the grid halved to a few cells, red-black Gauss–Seidel smoothing each level


class Carrier(IntEnum):
    """What carries the density: the grid's volume, or particles moved through the grid's velocity."""
    GRID = 0
    PARTICLES = 1


class ParticleDraw(IntEnum):
    """How the particles are drawn."""
    POINTS = 0              # one pixel a particle
    STREAKS = 1             # a line from the previous to the current position
    BLOBS = 2               # a gaussian a particle, the sum a soft melt
    SURFACE = 3             # spheres melted into one surface, shaded (screen-space fluid rendering)


class Overlap(IntEnum):
    """How the particles cover each other, for points, streaks and blobs."""
    OCCLUDED = 0            # depth slices marched front to back: near ink covers far ink
    ADDITIVE = 1            # one sum, the depth as fog per particle


class ParticleSettings(BaseSettings):
    """The particle carrier: the ring, the emission, the look."""
    count: Field[int]               = Field(2_000_000, min=10_000, max=8_000_000, step=10_000,     description="Particles in the ring, the most alive at once; 32 bytes each")
    rate: Field[float]              = Field(20.0,   min=0.01,   max=50.0,                               description="Particles per texel per second where ink is (fewer: heavier, bigger blobs)")
    ring_fade: Field[float]         = Field(0.1,    min=0.01,   max=1.0,                                description="The ring's last share the oldest fade over before they are overwritten")
    draw: Field[ParticleDraw]       = Field(ParticleDraw.POINTS,                                        description="Points, streaks, blobs or a surface")
    overlap: Field[Overlap]         = Field(Overlap.OCCLUDED,                                           description="Occluded: near ink covers far ink. Additive: one sum")
    slices: Field[int]              = Field(16,     min=2,      max=32,                                 description="Depth slices of the occluded look, at the output's size")
    size: Field[float]              = Field(1.0,    min=1.0,    max=8.0,                                description="A point's size in pixels")
    streak: Field[float]            = Field(1.0,    min=0.0,    max=4.0,                                description="A streak's length as a share of its step")
    radius: Field[float]            = Field(24.0,   min=1.0,    max=48.0,                               description="A blob's or a sphere's radius in output pixels at the reference mass")
    spread: Field[float]            = Field(2.0,    min=0.0,    max=10.0,                               description="A blob's widening with age (pixels² per second), at most twice its radius")
    smooth: Field[int]              = Field(3,      min=0,      max=8,                                  description="The surface's smoothing passes")


class BoundarySettings(BaseSettings):
    """The boundary: the body in the water, whose velocity the water is dragged toward."""
    gain: Field[float]              = Field(1.0,    min=0.0,    max=10.0,   color="teal",           description="The body's velocity times this is what the water is dragged toward; 1 = as fast as the body")
    drag_time: Field[float]         = Field(50.0,   min=0.0,    max=1000.0, step=1.0, color="cyan", description="Time the water takes to close the gap to the body's velocity (ms); 0 = at once")


class BodyForceSettings(BaseSettings):
    """The body force: the Navier–Stokes force term, an acceleration added × dt."""
    gain: Field[float]              = Field(1.0,    min=0.0,    max=10.0,   color="teal",           description="The acceleration times this is added × dt; 1 = its Δv added once")


class VelocitySettings(BaseSettings):
    """Velocity field parameters."""
    self_advection: Field[float]    = Field(1.0,    min=0.0,    max=1.0,    color="blue-grey",      description="How far the velocity carries itself per step; 1 = the full step, the physics")
    fade_time: Field[float]         = Field(60.0,   min=0.1,    max=60.0,   color="deep-blue",      description="Seconds until velocity fades to ~1%; 60 = water keeps its momentum")
    dampen_threshold: Field[float]  = Field(50.0,   min=0.1,    max=50.0,   color="cyan",           description="Speed above which velocity is dampened (widths/s); a guard far above any body")
    dampen_time: Field[float]       = Field(2.0,    min=0.0,    max=10.0,   color="light-blue",     description="Seconds for excess above threshold to decay to ~1%. 0=off")
    vorticity: Field[float]         = Field(0.3,    min=0.0,    max=5.0,    color="purple",         description="Vortex confinement's epsilon, its force epsilon × cell × (N × curl) (Fedkiw 2001)")
    vorticity_radius: Field[float]  = Field(1.0,    min=1.0,    max=8.0,    color="deep-purple",    description="Curl sampling radius in cells; 1 = the neighbors")
    viscosity: Field[float]         = Field(0.0,    min=0.0,    max=0.01,   step=0.0001, color="indigo", description="Kinematic viscosity (widths² per second); 0 = water at this scale")
    viscosity_iter: Field[int]      = Field(20,     min=1,      max=60,     color="orange",         description="Viscosity solver iterations at 60 fps; fewer at a higher rate, never more")


class DensitySettings(BaseSettings):
    """Density field parameters."""
    boundary_gain: Field[float]     = Field(1.0,    min=0.0,    max=10.0,   color="teal",           description="The boundary's density times this is added × dt; 1 = one unit per width of travel")
    body_force_gain: Field[float]   = Field(1.0,    min=0.0,    max=10.0,   color="teal",           description="The body force's density times this is added × dt; 1 = one unit per width/s of Δv")
    carrier: Field[Carrier]         = Field(Carrier.GRID,                                           description="What carries the density: the grid's volume or particles")
    size: Field[int]                = Field(1,      min=1,      max=4,                              description="The density grid's x and y size as a multiple of the simulation's; 1 = one grid")
    fade_time: Field[float]         = Field(60.0,   min=0.01,   max=60.0,   color="deep-blue",      description="Seconds until density fades to ~1%; 60 = ink stays")
    evaporation: Field[float]       = Field(0.0,    min=0.0,    max=2.0,    color="grey",           description="Density removed per second; clears low-level haze (0 = off)")
    wall: Field[WallMode]           = Field(WallMode.CLOSED,                                        description="Open: density absorbed at the walls. Closed: density reflects")


class TemperatureSettings(BaseSettings):
    """Temperature field parameters."""
    boundary_gain: Field[float]     = Field(1.0,    min=0.0,    max=10.0,   color="teal",           description="The boundary's temperature times this is added × dt, signed by the color")
    body_force_gain: Field[float]   = Field(1.0,    min=0.0,    max=10.0,   color="teal",           description="The body force's temperature times this is added × dt, signed by the color")
    fade_time: Field[float]         = Field(3.0,    min=0.01,   max=60.0,   color="deep-blue",      description="Seconds until temperature fades to ~1%")
    buoyancy: Field[float]          = Field(0.0,    min=0.0,    max=10.0,   color="deep-orange",    description="Lift per unit of temperature above ambient (widths/s²) against gravity; 0 = none")
    sink: Field[float]              = Field(0.0,    min=0.0,    max=20.0,   color="red",            description="Weight per unit of density (widths/s²) along gravity; 0 = as heavy as the water")
    gravity_x: Field[float]         = Field(0.0,    min=-1.0,   max=1.0,    color="deep-orange",    description="Gravity direction X")
    gravity_y: Field[float]         = Field(1.0,    min=-1.0,   max=1.0,    color="deep-orange",    description="Gravity direction Y; 1 = down the screen")
    gravity_z: Field[float]         = Field(0.0,    min=-1.0,   max=1.0,    color="deep-orange",    description="Gravity direction Z; 1 = away from viewer")
    ambient: Field[float]           = Field(0.0,    min=0.0,    max=1.0,    color="brown",          description="Reference temperature (buoyancy = 0 at this temp)")
    wall: Field[WallMode]           = Field(WallMode.CLOSED,                                        description="Closed: insulated walls (reflects). Open: cold walls (0 at the walls)")


class PressureSettings(BaseSettings):
    """Pressure field parameters."""
    solver: Field[PressureSolver]   = Field(PressureSolver.JACOBI,                                  description="Jacobi, red-black SOR (Gauss–Seidel over-relaxed) or multigrid V-cycles")
    iterations: Field[int]          = Field(40,     min=1,      max=60,     color="orange",         description="Pressure solver iterations per step; for SOR its red-black sweeps")
    relaxation: Field[float]        = Field(1.8,    min=1.0,    max=1.85,   color="orange",         description="SOR's over-relaxation factor omega; 1 = Gauss–Seidel; 1.95 diverged")
    cycles: Field[int]              = Field(1,      min=1,      max=8,      color="orange",         description="Multigrid V-cycles per step")
    wall: Field[WallMode]           = Field(WallMode.CLOSED,                                        description="The x and y walls. Closed: solid, the water cannot pass. Open: pressure escapes")


class VolumeSettings(BaseSettings):
    """The volume: the grid's z layers, where the body sits in it, and the composite's look."""
    layers: Field[int]              = Field(128,  min=1,    max=256,    description="z layers; the cells are cubic, so the slab is layers / width deep")
    injection_layer: Field[float]   = Field(0.5,  min=0.0,  max=1.0,    description="Normalized depth of the injection's center, where the body sits")
    injection_spread: Field[float]  = Field(0.075, min=0.001, max=0.2,  step=0.001, description="Gaussian sigma of the injection's depth (widths); 0.075 = a body 0.3 widths thick")
    emission: Field[float]          = Field(1.0,  min=0.0,  max=8.0,    color="amber",          description="Brightness gain per unit density")
    absorption: Field[float]        = Field(4.0,  min=0.01, max=50.0,   color="deep-orange",    description="Occlusion: extinction per unit density (near density covers far density)")
    glow: Field[float]              = Field(0.25, min=0.0,  max=1.0,    color="yellow",         description="Additive glow mix: 1 = additive (depth invisible)")
    tint: Field[float]              = Field(0.5,  min=0.0,  max=1.0,    color="light-blue",     description="Far density shifts toward water color (red absorbed first)")
    fog: Field[float]               = Field(0.5,  min=0.0,  max=4.0,    color="blue-grey",      description="Water extinction over the full depth; dims far layers")
    wall: Field[WallMode]           = Field(WallMode.CLOSED,                                        description="The z walls. Closed: solid, front and back reflect. Open: they absorb")


class Fluid3DSettings(BaseSettings):
    """The 3D fluid's configuration.

    Speed model:
        One 'speed' parameter controls the passive scalar transport (density and temperature).
        At speed=1.0, a velocity value of 1.0 moves fluid across the full texture width in 1 second.

    Fade_time model:
        Exponential frame-rate-independent decay: multiplier = 0.01^(dt/fade_time).
        fade_time=3.0 means the field retains ~1% after 3 seconds.
    """

    # ---- Grid dimensions (multiples of 32) ----
    width: Field[int]               = Field(256,    min=32,     max=4096,   step=32,    description="Simulation grid width")
    height: Field[int]              = Field(448,    min=32,     max=4096,   step=32,    description="Simulation grid height")
    time_step: Field[TimeStepMode]  = Field(TimeStepMode.MEASURED,                  description="Step by the measured render rate, or by a fixed rate")
    fixed_fps: Field[float]         = Field(60.0,   min=1.0,    max=240.0,              description="Steps per second in fixed mode (fps)")
    fps_smoothing: Field[float]     = Field(1.0,    min=0.0,    max=10.0,               description="Smoothing of the measured rate in measured mode (s); 0 follows each frame")
    speed: Field[float]             = Field(1.0,    min=0.0,    max=5.0,    description="Base fluid transport rate")
    advection: Field[Advection]     = Field(Advection.SEMI_LAGRANGIAN,              description="How the fields are carried: MacCormack keeps swirls and edges, at about twice the reads")

    # ---- The inputs, then the field groups ----
    boundary: Group[BoundarySettings]       = Group(BoundarySettings)
    body_force: Group[BodyForceSettings]    = Group(BodyForceSettings)
    volume: Group[VolumeSettings]           = Group(VolumeSettings)
    velocity: Group[VelocitySettings]       = Group(VelocitySettings)
    density: Group[DensitySettings]         = Group(DensitySettings)
    particles: Group[ParticleSettings]      = Group(ParticleSettings)
    temperature: Group[TemperatureSettings] = Group(TemperatureSettings)
    pressure: Group[PressureSettings]       = Group(PressureSettings)
