"""The 2D fluid's settings: one group per field (velocity, density, temperature, pressure), with
the field's name as the group's, so the field names carry no prefix."""

from modules.settings import Field, BaseSettings, Group

from ..boundary_mode import BoundaryMode
from ..time_step import TimeStepMode


class VelocitySettings(BaseSettings):
    """Velocity field parameters."""
    gain: Field[float]              = Field(1.0,    min=0.0,    max=10.0,   color="teal",           description="Multiplier applied to the measured (x, y) flow input")
    self_advection: Field[float]    = Field(0.01,   min=0.0,    max=1.0,    color="blue-grey",      description="How much velocity advects itself. Keep low for stability.")
    fade_time: Field[float]         = Field(30.0,   min=0.1,    max=60.0,   color="deep-blue",      description="Seconds until velocity fades to ~1%")
    dampen_threshold: Field[float]  = Field(5.0,    min=0.1,    max=50.0,   color="cyan",           description="Magnitude above which velocity is dampened")
    dampen_time: Field[float]       = Field(0.5,    min=0.0,    max=10.0,   color="light-blue",     description="Seconds for excess above threshold to decay to ~1%. 0=off")
    vorticity: Field[float]         = Field(5.17,   min=0.0,    max=60.0,   color="purple",         description="Vortex confinement strength (adds turbulence)")
    vorticity_radius: Field[float]  = Field(2.82,   min=1.0,    max=30.0,   color="deep-purple",    description="Curl sampling radius in texels")
    viscosity: Field[float]         = Field(6.62,   min=0.0,    max=100.0,  color="indigo",         description="Fluid thickness/resistance to flow")
    viscosity_iter: Field[int]      = Field(40,     min=1,      max=60,     color="orange",         description="Solver quality for viscosity (iterations at 60fps, auto-scaled for frame rate)")


class DensitySettings(BaseSettings):
    """Density field parameters."""
    gain: Field[float]              = Field(1.0,    min=0.0,    max=10.0,    color="teal",          description="Multiplier applied to all density inputs")
    speed_offset: Field[float]      = Field(0.0,    min=-5.0,   max=5.0,    color="blue-grey",      description="Added to base speed for density only. 0 = physically coupled.")
    fade_time: Field[float]         = Field(4.0,    min=0.01,   max=60.0,   color="deep-blue",      description="Seconds until density fades to ~1%")
    dampen_threshold: Field[float]  = Field(1.2,    min=0.1,    max=5.0,    color="cyan",           description="Magnitude above which density is dampened")
    dampen_time: Field[float]       = Field(0.5,    min=0.0,    max=10.0,   color="light-blue",     description="Seconds for excess above threshold to decay to ~1%. 0=off")
    boundary: Field[BoundaryMode]   = Field(BoundaryMode.OPEN,                                      description="Open: density absorbed at edges. Closed: density reflects at edges")


class TemperatureSettings(BaseSettings):
    """Temperature field parameters."""
    gain: Field[float]              = Field(0.0,    min=0.0,    max=2.0,    color="teal",           description="Multiplier applied to all temperature inputs")
    speed_offset: Field[float]      = Field(0.0,    min=-5.0,   max=5.0,    color="blue-grey",      description="Added to base speed for temperature only. 0 = physically coupled.")
    fade_time: Field[float]         = Field(3.0,    min=0.01,   max=60.0,   color="deep-blue",      description="Seconds until temperature fades to ~1%")
    buoyancy: Field[float]          = Field(0.0,    min=0.0,    max=10.0,   color="deep-orange",    description="Thermal buoyancy coefficient: hot (fresh) density rises against gravity")
    sink: Field[float]              = Field(4.0,    min=0.0,    max=20.0,   color="red",            description="Weight of the density: it sinks along gravity")
    ambient: Field[float]           = Field(0.2,    min=0.0,    max=1.0,    color="brown",          description="Reference temperature (buoyancy = 0 at this temp)")
    boundary: Field[BoundaryMode]   = Field(BoundaryMode.CLOSED,                                    description="Closed: insulated walls (reflects). Open: cold walls (0 at edges)")


class PressureSettings(BaseSettings):
    """Pressure field parameters."""
    speed: Field[float]             = Field(0.0,    min=0.0,    max=2.0,    color="blue-grey",      description="Pressure advection speed")
    fade_time: Field[float]         = Field(8.0,    min=0.01,   max=60.0,   color="deep-blue",      description="Seconds until pressure fades to ~1%")
    iterations: Field[int]          = Field(40,     min=1,      max=60,     color="orange",         description="Solver quality for pressure (iterations at 60fps, auto-scaled for frame rate)")
    boundary: Field[BoundaryMode]   = Field(BoundaryMode.OPEN,                                      description="Open: pressure escapes at edges (less sticky). Closed: pressure reflects")


class Fluid2DSettings(BaseSettings):
    """The 2D fluid's configuration.

    Speed model:
        One 'speed' parameter controls all passive scalar transport (density,
        temperature, pressure). At speed=1.0, a velocity value of 1.0 moves
        fluid across the full texture width in 1 second.

    Fade_time model:
        Exponential frame-rate-independent decay: multiplier = 0.01^(dt/fade_time).
        fade_time=3.0 means the field retains ~1% after 3 seconds.
    """

    # ---- Profiling ----
    profile: Field[bool] = Field(False, description="Enable per-pass GPU timing (uses glFinish; serializes pipeline)")

    # ---- Grid dimensions (multiples of 32) ----
    width: Field[int]               = Field(1024,   min=32,     max=4096,   step=32,    description="Simulation grid width")
    height: Field[int]              = Field(576,    min=32,     max=4096,   step=32,    description="Simulation grid height")
    time_step: Field[TimeStepMode]  = Field(TimeStepMode.MEASURED,                  description="Step by the measured render rate, or by a fixed rate")
    fixed_fps: Field[float]         = Field(60.0,   min=1.0,    max=240.0,              description="Steps per second in fixed mode (fps)")
    fps_smoothing: Field[float]     = Field(1.0,    min=0.0,    max=10.0,               description="Smoothing of the measured rate in measured mode (s); 0 follows each frame")
    speed: Field[float]             = Field(1.0,    min=0.0,    max=5.0,    description="Base fluid transport rate")

    # ---- Field groups ----
    velocity: Group[VelocitySettings]       = Group(VelocitySettings)
    density: Group[DensitySettings]         = Group(DensitySettings)
    temperature: Group[TemperatureSettings] = Group(TemperatureSettings)
    pressure: Group[PressureSettings]       = Group(PressureSettings)
