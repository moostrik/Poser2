"""The 3D fluid's settings: one group per field (velocity, density, temperature, pressure) and
the volume, with the field's name as the group's, so the field names carry no prefix."""

from modules.settings import Field, BaseSettings, Group

from ..wall_mode import WallMode
from ..time_step import TimeStepMode


class VelocitySettings(BaseSettings):
    """Velocity field parameters."""
    gain: Field[float]              = Field(1.0,    min=0.0,    max=10.0,   color="teal",           description="Multiplier applied to the measured (x, y, z) velocity input")
    self_advection: Field[float]    = Field(0.01,   min=0.0,    max=1.0,    color="blue-grey",      description="How much velocity advects itself. Keep low for stability.")
    fade_time: Field[float]         = Field(30.0,   min=0.1,    max=60.0,   color="deep-blue",      description="Seconds until velocity fades to ~1%")
    dampen_threshold: Field[float]  = Field(5.0,    min=0.1,    max=50.0,   color="cyan",           description="Magnitude above which velocity is dampened")
    dampen_time: Field[float]       = Field(0.5,    min=0.0,    max=10.0,   color="light-blue",     description="Seconds for excess above threshold to decay to ~1%. 0=off")
    vorticity: Field[float]         = Field(5.17,   min=0.0,    max=60.0,   color="purple",         description="Vortex confinement strength (adds turbulence)")
    vorticity_radius: Field[float]  = Field(2.82,   min=1.0,    max=30.0,   color="deep-purple",    description="Curl sampling radius in texels")
    viscosity: Field[float]         = Field(6.62,   min=0.0,    max=100.0,  color="indigo",         description="Fluid thickness/resistance to flow")
    viscosity_iter: Field[int]      = Field(40,     min=1,      max=60,     color="orange",         description="Viscosity solver iterations at 60 fps; fewer at a higher rate, never more")


class DensitySettings(BaseSettings):
    """Density field parameters."""
    gain: Field[float]              = Field(1.0,    min=0.0,    max=10.0,    color="teal",          description="Multiplier applied to all density inputs")
    fade_time: Field[float]         = Field(4.0,    min=0.01,   max=60.0,   color="deep-blue",      description="Seconds until density fades to ~1%")
    evaporation: Field[float]       = Field(0.0,    min=0.0,    max=2.0,    color="grey",           description="Density removed per second; clears low-level haze (0 = off)")
    wall: Field[WallMode]           = Field(WallMode.OPEN,                                          description="Open: density absorbed at the walls. Closed: density reflects")


class TemperatureSettings(BaseSettings):
    """Temperature field parameters."""
    gain: Field[float]              = Field(0.0,    min=0.0,    max=2.0,    color="teal",           description="Multiplier applied to all temperature inputs")
    fade_time: Field[float]         = Field(3.0,    min=0.01,   max=60.0,   color="deep-blue",      description="Seconds until temperature fades to ~1%")
    buoyancy: Field[float]          = Field(0.0,    min=0.0,    max=10.0,   color="deep-orange",    description="Thermal buoyancy coefficient: hot (fresh) density rises against gravity")
    sink: Field[float]              = Field(4.0,    min=0.0,    max=20.0,   color="red",            description="Weight of the density: it sinks along gravity")
    gravity_x: Field[float]         = Field(0.0,    min=-1.0,   max=1.0,    color="deep-orange",    description="Gravity direction X")
    gravity_y: Field[float]         = Field(1.0,    min=-1.0,   max=1.0,    color="deep-orange",    description="Gravity direction Y; 1 = down the screen")
    gravity_z: Field[float]         = Field(0.0,    min=-1.0,   max=1.0,    color="deep-orange",    description="Gravity direction Z; 1 = away from viewer")
    ambient: Field[float]           = Field(0.2,    min=0.0,    max=1.0,    color="brown",          description="Reference temperature (buoyancy = 0 at this temp)")
    wall: Field[WallMode]           = Field(WallMode.CLOSED,                                        description="Closed: insulated walls (reflects). Open: cold walls (0 at the walls)")


class PressureSettings(BaseSettings):
    """Pressure field parameters."""
    iterations: Field[int]          = Field(40,     min=1,      max=60,     color="orange",         description="Pressure solver iterations per step")
    wall: Field[WallMode]           = Field(WallMode.OPEN,                                          description="Open: pressure escapes at the walls (less sticky). Closed: pressure reflects")


class VolumeSettings(BaseSettings):
    """The volume: the 3D fluid's z layers and the composite's look."""
    layers: Field[int]              = Field(4,    min=1,    max=64,     description="Density z layers (the stored density volume)")
    sim_layers: Field[int]          = Field(4,    min=1,    max=128,    description="Velocity/pressure/temperature z layers (the sim grid)")
    spacing: Field[float]           = Field(4.0,  min=1.0,  max=64.0,   description="Z layer spacing in sim texels (low = lively depth, high = flat)")
    emission: Field[float]          = Field(1.0,  min=0.0,  max=8.0,    color="amber",          description="Brightness gain per unit density")
    gamma: Field[float]             = Field(1.0,  min=0.5,  max=3.0,    color="orange",         description="Density response exponent: >1 hides thin haze, pops dense cores")
    absorption: Field[float]        = Field(4.0,  min=0.01, max=50.0,   color="deep-orange",    description="Occlusion: extinction per unit density (near density covers far density)")
    glow: Field[float]              = Field(0.25, min=0.0,  max=1.0,    color="yellow",         description="Additive glow mix: 1 = legacy additive (depth invisible)")
    tint: Field[float]              = Field(0.5,  min=0.0,  max=1.0,    color="light-blue",     description="Far density shifts toward water color (red absorbed first)")
    fog: Field[float]               = Field(0.5,  min=0.0,  max=4.0,    color="blue-grey",      description="Water extinction over the full depth; dims far layers")
    parallax: Field[float]          = Field(0.02, min=0.0,  max=0.1,    color="purple",         description="Virtual eye offset in UV units; layers shear with depth")
    perspective: Field[float]       = Field(0.15, min=0.0,  max=1.0,    color="deep-purple",    description="Per-layer shrink toward the vanishing point; 0 = flat")
    drift_period: Field[float]      = Field(45.0, min=0.0,  max=180.0,  color="indigo",         description="Seconds per orbit of the drifting eye; 0 = static offset")
    eye_tracking: Field[float]      = Field(0.0,  min=0.0,  max=1.0,    color="violet",         description="Parallax eye follows the tracked head; 0 = autonomous drift")
    layer_blur: Field[float]        = Field(0.0,  min=0.0,  max=8.0,    color="brown",          description="Blur radius in texels at the back layer; 0 = off")
    debug_layers: Field[bool]       = Field(False,                                              description="Show the dominant z layer as hue (red = back, violet = front)")
    injection_layer: Field[float]   = Field(1.0,  min=0.0,  max=1.0,    description="Normalized depth for 2D->3D injection center")
    injection_spread: Field[float]  = Field(0.001, min=0.001, max=0.5,   description="Gaussian sigma for depth spread during injection")
    wall: Field[WallMode]           = Field(WallMode.CLOSED,                                        description="The z walls. Closed: front and back reflect. Open: they absorb")


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
    width: Field[int]               = Field(1024,   min=32,     max=4096,   step=32,    description="Simulation grid width")
    height: Field[int]              = Field(576,    min=32,     max=4096,   step=32,    description="Simulation grid height")
    time_step: Field[TimeStepMode]  = Field(TimeStepMode.MEASURED,                  description="Step by the measured render rate, or by a fixed rate")
    fixed_fps: Field[float]         = Field(60.0,   min=1.0,    max=240.0,              description="Steps per second in fixed mode (fps)")
    fps_smoothing: Field[float]     = Field(1.0,    min=0.0,    max=10.0,               description="Smoothing of the measured rate in measured mode (s); 0 follows each frame")
    speed: Field[float]             = Field(1.0,    min=0.0,    max=5.0,    description="Base fluid transport rate")

    # ---- Field groups ----
    volume: Group[VolumeSettings]           = Group(VolumeSettings)
    velocity: Group[VelocitySettings]       = Group(VelocitySettings)
    density: Group[DensitySettings]         = Group(DensitySettings)
    temperature: Group[TemperatureSettings] = Group(TemperatureSettings)
    pressure: Group[PressureSettings]       = Group(PressureSettings)
