"""The 2D fluid's settings: the bridge's input, then one group per field (velocity, density, temperature, the
expansion, pressure), with the field's name as the group's, so the field names carry no prefix."""

from modules.settings import Field, BaseSettings, Group

from ..density_blend import DensityBlend
from ..port import ColorSpace
from ..wall_mode import WallMode
from ..time_step import TimeStepMode


class InputSettings(BaseSettings):
    """The bridge's input, in the order the layer injects it, as the 3D fluid takes it: the boundary's velocity and
    temperature, each the water dragged toward the body's value times a gain (capped at ±1 warm) over a time; the
    body force's acceleration added × dt times a gain; the density, its dye over the water's, added, or the water
    drawn toward it."""
    velocity_drag: Field[float]     = Field(50.0,   min=0.0,    max=1000.0, step=1.0, color="cyan", description="Time the water takes to close the gap to the body's velocity (ms); 0 = at once")
    velocity_gain: Field[float]     = Field(1.0,    min=-10.0,  max=10.0,   color="teal",           description="The body's velocity times this is what the water is dragged toward; 1 = as fast as the body")
    acceleration_gain: Field[float] = Field(1.0,    min=-10.0,  max=10.0,   color="teal",           description="The body force's acceleration times this is added × dt; 1 = its Δv added once")
    density_drag: Field[float]      = Field(200.0,  min=0.0,    max=1000.0, step=1.0, color="cyan", newline=True, description="Over, fill: time the water takes to take on the body's dye (ms); 0 = at once")
    density_release: Field[float]   = Field(0.0,    min=0.0,    max=1000.0, step=1.0, color="cyan", description="Time the water's dye takes to fall to the body's where it is less (ms); 0 = never falls")
    density_gain: Field[float]      = Field(1.0,    min=0.0,    max=10.0,   color="teal",           description="The body's amount times this; over, fill: at most fully dyed; 1 = fully dyed at 1 width/s")
    density_blend: Field[DensityBlend] = Field(DensityBlend.OVER,                                  description="The body's dye over the water's, added to it, or the water drawn toward it")
    temperature_drag: Field[float]  = Field(200.0,  min=0.0,    max=1000.0, step=1.0, color="cyan", newline=True, description="Time the water takes to take on the body's warmth (ms); 0 = at once")
    temperature_gain: Field[float]  = Field(1.0,    min=-10.0,  max=10.0,   color="teal",           description="The body's presence times this, at most ±1, is its warmth; below 0 cold; 1 = fully warm at 1 width/s")


class VelocitySettings(BaseSettings):
    """Velocity field parameters."""
    self_advection: Field[float]    = Field(0.01,   min=0.0,    max=1.0,    color="blue-grey",      description="How much velocity advects itself. Keep low for stability.")
    fade_time: Field[float]         = Field(30.0,   min=0.1,    max=60.0,   color="deep-blue",      description="Seconds until velocity fades to ~1%")
    dampen_threshold: Field[float]  = Field(5.0,    min=0.1,    max=50.0,   color="cyan",           description="Magnitude above which velocity is dampened")
    dampen_time: Field[float]       = Field(0.5,    min=0.0,    max=10.0,   color="light-blue",     description="Seconds for excess above threshold to decay to ~1%. 0=off")
    vorticity: Field[float]         = Field(5.17,   min=0.0,    max=60.0,   color="purple",         description="Vortex confinement strength (adds turbulence)")
    vorticity_radius: Field[float]  = Field(2.82,   min=1.0,    max=30.0,   color="deep-purple",    description="Curl sampling radius in texels")
    viscosity: Field[float]         = Field(6.62,   min=0.0,    max=100.0,  color="indigo",         description="Fluid thickness/resistance to flow")
    viscosity_iter: Field[int]      = Field(40,     min=1,      max=60,     color="orange",         description="Viscosity solver iterations at 60 fps; fewer at a higher rate, never more")


class DensitySettings(BaseSettings):
    """Density field parameters: the density held premultiplied, its color times its amount, and its amount."""
    color_space: Field[ColorSpace]  = Field(ColorSpace.RGB,                                         description="The color stored as RGB, or as Oklab, mixing as the eye sees; a switch clears the density")
    speed_offset: Field[float]      = Field(0.0,    min=-5.0,   max=5.0,    color="blue-grey",      description="Added to base speed for density only. 0 = physically coupled.")
    fade_time: Field[float]         = Field(4.0,    min=0.01,   max=60.0,   color="deep-blue",      description="Seconds until density fades to ~1%")
    dampen_threshold: Field[float]  = Field(1.2,    min=0.1,    max=5.0,    color="cyan",           description="Magnitude above which density is dampened")
    dampen_time: Field[float]       = Field(0.5,    min=0.0,    max=10.0,   color="light-blue",     description="Seconds for excess above threshold to decay to ~1%. 0=off")
    wall: Field[WallMode]           = Field(WallMode.OPEN,                                          description="Open: density absorbed at the walls. Closed: density reflects")


class TemperatureSettings(BaseSettings):
    """Temperature field parameters."""
    speed_offset: Field[float]      = Field(0.0,    min=-5.0,   max=5.0,    color="blue-grey",      description="Added to base speed for temperature only. 0 = physically coupled.")
    fade_time: Field[float]         = Field(3.0,    min=0.01,   max=60.0,   color="deep-blue",      description="Seconds until temperature fades to ~1%")
    buoyancy: Field[float]          = Field(0.0,    min=0.0,    max=10.0,   color="deep-orange",    description="Thermal buoyancy coefficient: hot (fresh) density rises against gravity")
    sink: Field[float]              = Field(4.0,    min=0.0,    max=20.0,   color="red",            description="Weight of the density: it sinks along gravity")
    ambient: Field[float]           = Field(0.2,    min=0.0,    max=1.0,    color="brown",          description="Reference temperature (buoyancy = 0 at this temp)")
    wall: Field[WallMode]           = Field(WallMode.CLOSED,                                        description="Closed: insulated walls (reflects). Open: cold walls (0 at the walls)")


class ExpansionSettings(BaseSettings):
    """The expansion: the water swelling where the body is, the divergence the projection makes (Nguyen, Fedkiw
    and Jensen 2002)."""
    enabled: Field[bool]            = Field(False,                                      description="The expansion on while its rate is not 0; off, skipped")
    rate: Field[float]              = Field(0.0,    min=-5.0,   max=5.0,    step=0.01, color="teal", description="Volume the water gains per second where the body is (1/s); below 0 it shrinks; 0 = none")


class PressureSettings(BaseSettings):
    """Pressure field parameters."""
    speed: Field[float]             = Field(0.0,    min=0.0,    max=2.0,    color="blue-grey",      description="Pressure advection speed")
    fade_time: Field[float]         = Field(8.0,    min=0.01,   max=60.0,   color="deep-blue",      description="Seconds until pressure fades to ~1%")
    iterations: Field[int]          = Field(40,     min=1,      max=60,     color="orange",         description="Pressure solver iterations per step")
    wall: Field[WallMode]           = Field(WallMode.OPEN,                                          description="Open: pressure escapes at the walls (less sticky). Closed: pressure reflects")


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

    # ---- The input, then the field groups ----
    input: Group[InputSettings]             = Group(InputSettings)
    velocity: Group[VelocitySettings]       = Group(VelocitySettings)
    density: Group[DensitySettings]         = Group(DensitySettings)
    temperature: Group[TemperatureSettings] = Group(TemperatureSettings)
    expansion: Group[ExpansionSettings]     = Group(ExpansionSettings)
    pressure: Group[PressureSettings]       = Group(PressureSettings)
