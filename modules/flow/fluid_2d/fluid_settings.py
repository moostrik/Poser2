"""The 2D fluid's settings, as the 3D's for what a plane has: the grid, then one group per concern (time, input,
velocity, viscosity, vorticity, temperature, expansion, pressure, density), with the concern's name as the group's,
so the field names carry no prefix."""

from modules.settings import Field, BaseSettings, Group, Widget

from ..advection import Advection, Limiter
from ..density_blend import DensityBlend
from ..port import ColorSpace
from ..pressure_solver import PressureSolver
from ..wall_mode import WallMode
from ..time_step import TimeStepMode


class TimeSettings(BaseSettings):
    """The fluid's time: how a step's length is taken, and the fluid's time against real time."""
    time_step: Field[TimeStepMode]  = Field(TimeStepMode.MEASURED,                                  description="Step by the measured render rate, or by a fixed rate")
    fixed_fps: Field[float]         = Field(60.0,   min=1.0,    max=240.0,                          description="Steps per second in fixed mode (fps)")
    fps_smoothing: Field[float]     = Field(1.0,    min=0.0,    max=10.0,                           description="Smoothing of the measured rate in measured mode (s); 0 follows each frame")
    time_scale: Field[float]        = Field(1.0,    min=0.05,   max=2.0,    step=0.05,              description="The fluid's time as a share of real time; 0.5 = everything at half speed, 1 = real time")


class GravitySettings(BaseSettings):
    """Gravity's direction, which the density's weight (`density.sink`) and the warmth's lift (`temperature.buoyancy`)
    act along; its length scales both."""
    x: Field[float]                 = Field(0.0,    min=-1.0,   max=1.0,    color="deep-orange",    description="Gravity's direction, x right")
    y: Field[float]                 = Field(-1.0,   min=-1.0,   max=1.0,    color="deep-orange",    description="Gravity's direction, y up; -1 = down the screen")


class InputSettings(BaseSettings):
    """The bridge's input, in the order the layer injects it, as the 3D fluid takes it: the boundary's velocity and
    temperature, each the water dragged toward the body's value times a gain (capped at ±1 warm) over a time; the
    body force's acceleration added × dt times a gain; the density, its dye over the water's, added, or the water drawn toward it."""
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
    """The velocity: how it carries itself and fades."""
    advection: Field[Advection]     = Field(Advection.SEMI_LAGRANGIAN,                  description="How the velocity carries itself: MacCormack keeps swirls, at about twice the reads")
    limiter: Field[Limiter]         = Field(Limiter.CLAMP,                              description="Where MacCormack overshoots: clamped to the step's bounds, or the plain step")
    advection_scale: Field[float]   = Field(1.0,    min=0.0,    max=1.0,    color="blue-grey",      description="How far the velocity carries itself per step; 1 = with the water, the physics")
    fade_time: Field[float]         = Field(0.0,    min=0.0,    max=300.0,  step=1.0, color="deep-blue", description="Seconds until the velocity fades to 1%, a share at a time; 0 = off")
    evaporation_time: Field[float]  = Field(0.0,    min=0.0,    max=300.0,  step=1.0, color="grey", description="Seconds 1 width/s takes to brake well above the knee, a saturating loss; 0 = off")
    evaporation_knee: Field[float]  = Field(0.1,    min=0.001,  max=1.0,    widget=Widget.log_slider, color="grey", description="Speed (widths/s) below which the braking turns from a fixed amount to a share")


class ViscositySettings(BaseSettings):
    """The viscosity: the velocity's implicit diffusion, after its advection."""
    enabled: Field[bool]            = Field(False,                                      description="The viscosity on while it is above 0; off, skipped")
    viscosity: Field[float]         = Field(0.0,    min=0.0,    max=0.01,   step=0.0001, color="indigo", description="Kinematic viscosity (widths² per second); 0 = water at this scale")
    iterations: Field[int]          = Field(20,     min=1,      max=60,     color="orange",         description="Solver iterations at 60 fps; fewer at a higher rate, never more")


class VorticitySettings(BaseSettings):
    """The vorticity confinements (Fedkiw 2001), each keeping swirls of one size alive: the strengths on a row,
    each radius under its strength."""
    enabled: Field[bool]      = Field(False,                                      description="The confinements on, each while its strength is above 0; off, skipped")
    strength_1: Field[float]  = Field(0.0,    min=0.0,    max=100.0,  color="purple", newline=True, description="Confinement 1's epsilon, the force epsilon × cell × (N × curl); 0 = off")
    strength_2: Field[float]  = Field(0.0,    min=0.0,    max=100.0,  color="purple",         description="Confinement 2's epsilon; 0 = off")
    strength_3: Field[float]  = Field(0.0,    min=0.0,    max=100.0,  color="purple",         description="Confinement 3's epsilon; 0 = off")
    radius_1: Field[float]    = Field(0.004,  min=0.001,  max=0.1,    newline=True, widget=Widget.log_slider, color="deep-purple", description="The size of the swirls confinement 1 keeps, widths; at least one cell")
    radius_2: Field[float]    = Field(0.004,  min=0.001,  max=0.1,    widget=Widget.log_slider, color="deep-purple", description="The size of the swirls confinement 2 keeps, widths")
    radius_3: Field[float]    = Field(0.004,  min=0.001,  max=0.1,    widget=Widget.log_slider, color="deep-purple", description="The size of the swirls confinement 3 keeps, widths")


class TemperatureSettings(BaseSettings):
    """The temperature: its lift and how it cools; carried as the density is."""
    enabled: Field[bool]            = Field(False,                                      description="The warmth on: injected, carried and lifting while buoyancy is above 0; off, skipped")
    buoyancy: Field[float]          = Field(0.0,    min=0.0,    max=10.0,   color="deep-orange",    description="Lift per unit of warmth (widths/s²) against gravity; 0 = none")
    fade_time: Field[float]         = Field(0.0,    min=0.0,    max=300.0,  step=1.0, color="deep-blue", newline=True, description="Seconds until the warmth fades to 1%, a share at a time; 0 = off")
    evaporation_time: Field[float]  = Field(0.0,    min=0.0,    max=300.0,  step=1.0, color="grey", description="Seconds full warmth (±1) takes to fall well above the knee, a saturating loss; 0 = off")
    evaporation_knee: Field[float]  = Field(0.1,    min=0.001,  max=1.0,    widget=Widget.log_slider, color="grey", description="Warmth below which the fall turns from a fixed amount to a share")


class ExpansionSettings(BaseSettings):
    """The expansion: the water swelling where the body is, the divergence the projection makes (Nguyen, Fedkiw
    and Jensen 2002)."""
    enabled: Field[bool]            = Field(False,                                      description="The expansion on while its rate is not 0; off, skipped")
    rate: Field[float]              = Field(0.0,    min=-5.0,   max=5.0,    step=0.01, color="teal", description="Volume the water gains per second where the body is (1/s); below 0 it shrinks; 0 = none")


class PressureSettings(BaseSettings):
    """The pressure's solve, by the solver picked."""
    solver: Field[PressureSolver]   = Field(PressureSolver.JACOBI,                                  description="Jacobi, red-black SOR (Gauss–Seidel over-relaxed) or multigrid V-cycles")
    iterations: Field[int]          = Field(40,     min=1,      max=60,     color="orange",         description="Pressure solver iterations per step; for SOR its red-black sweeps")
    relaxation: Field[float]        = Field(1.8,    min=1.0,    max=1.85,   color="orange",         description="SOR's over-relaxation factor omega; 1 = Gauss–Seidel; 1.95 diverged")
    cycles: Field[int]              = Field(1,      min=1,      max=8,      color="orange",         description="Multigrid V-cycles per step")


class DensitySettings(BaseSettings):
    """The density: its color space, its weight, how it and the temperature are carried, how it fades and
    evaporates. Its size is the output's."""
    color_space: Field[ColorSpace]  = Field(ColorSpace.RGB,                                         description="The color stored as RGB, or as Oklab, mixing as the eye sees; a switch clears the density")
    sink: Field[float]              = Field(0.0,    min=0.0,    max=20.0,   color="red", newline=True, description="Weight per unit of amount (widths/s²) along gravity; 0 = as heavy as the water")
    advection: Field[Advection]     = Field(Advection.SEMI_LAGRANGIAN,  newline=True,   description="How the density and the temperature are carried: MacCormack keeps edges sharp")
    limiter: Field[Limiter]         = Field(Limiter.CLAMP,                              description="Where MacCormack overshoots: clamped to the step's bounds, or the plain step")
    advection_scale: Field[float]   = Field(1.0,    min=0.0,    max=5.0,    color="blue-grey",      description="How far the density and the temperature are carried per step; 1 = with the water")
    fade_time: Field[float]         = Field(0.0,    min=0.0,    max=300.0,  step=1.0, color="deep-blue", description="Seconds until the density fades to 1%, a share at a time; 0 = off")
    evaporation_time: Field[float]  = Field(0.0,    min=0.0,    max=300.0,  step=1.0, color="grey", description="Seconds fully dyed water takes to clear well above the knee, a saturating loss; 0 = off")
    evaporation_knee: Field[float]  = Field(0.1,    min=0.001,  max=1.0,    widget=Widget.log_slider, color="grey", description="Dye (0 to 1) below which the clearing turns from a fixed amount to a share")


class Fluid2DSettings(BaseSettings):
    """The 2D fluid's configuration: the grid, then one group per concern.

    Fade_time model:
        Exponential frame-rate-independent decay: multiplier = 0.01^(dt/fade_time), 0 = off.
        fade_time=3.0 means the field retains ~1% after 3 seconds. The evaporation is a saturating loss:
        dρ/dt = −ρ / (evaporation_time × (ρ + evaporation_knee)) on a field's size ρ (fully dyed, ±1 warm, 1 width/s).
    """

    # ---- The guard, how the fields are walled and slip, then the grid (multiples of 8) ----
    guard: Field[bool]              = Field(True,                                       description="Set non-finite values to 0, count and log them; off once the fluid is stable")
    wall: Field[WallMode]           = Field(WallMode.CLOSED,                            newline=True, description="Every side of every field. Closed: solid, the water slides along. Open: it leaves")
    slip_time: Field[float]         = Field(0.0,    min=0.0,    max=60.0,   step=0.1,   description="Seconds the water along a closed wall takes to slow to 1%; 0 = free-slip")
    width: Field[int]               = Field(1080,   min=32,     max=4096,   step=8,     newline=True, description="Simulation grid width")
    height: Field[int]              = Field(1920,   min=32,     max=4096,   step=8,     description="Simulation grid height")

    # ---- The groups ----
    time: Group[TimeSettings]               = Group(TimeSettings)
    gravity: Group[GravitySettings]         = Group(GravitySettings)
    input: Group[InputSettings]             = Group(InputSettings)
    # the fluid's passes in their order: the velocity, its viscosity, its confinement, the temperature's buoyancy,
    # the projection with the expansion, the temperature and the density carried
    velocity: Group[VelocitySettings]       = Group(VelocitySettings)
    viscosity: Group[ViscositySettings]     = Group(ViscositySettings)
    vorticity: Group[VorticitySettings]     = Group(VorticitySettings)
    temperature: Group[TemperatureSettings] = Group(TemperatureSettings)
    expansion: Group[ExpansionSettings]     = Group(ExpansionSettings)
    pressure: Group[PressureSettings]       = Group(PressureSettings)
    density: Group[DensitySettings]         = Group(DensitySettings)
