"""Shared configuration sub-groups for 2D and 3D fluid simulations.

Groups fields by physical domain (velocity, density, temperature, pressure)
with prefixes stripped — the group name provides context.
"""

from enum import IntEnum

from modules.settings import Field, BaseSettings, Widget, Group


class BoundaryMode(IntEnum):
    OPEN   = 0  # Dirichlet — zero value at boundary (field escapes / absorbing)
    CLOSED = 1  # Neumann   — zero gradient at boundary (field reflects / insulating)


class VelocitySettings(BaseSettings):
    """Velocity field parameters."""
    input_strength: Field[float]    = Field(1.0,    min=0.0,    max=10.0,   color="teal",           description="Multiplier applied to the measured (x, y) flow input")
    squeeze: Field[float]           = Field(0.0,    min=-10.0,  max=10.0,   color="cyan",           description="W synthesis: flow convergence; positive: looming pushes ink toward viewer (3D only)")
    push: Field[float]              = Field(0.0,    min=-2.0,   max=2.0,    color="light-blue",     description="W synthesis: flow magnitude; positive = away from viewer (3D only)")
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
    input_strength: Field[float]    = Field(1.0,    min=0.0,    max=10.0,    color="teal",          description="Multiplier applied to all density inputs")
    speed_offset: Field[float]      = Field(0.0,    min=-5.0,   max=5.0,    color="blue-grey",      description="Added to base speed for density only. 0 = physically coupled.")
    fade_time: Field[float]         = Field(4.0,    min=0.01,   max=60.0,   color="deep-blue",      description="Seconds until density fades to ~1%")
    evaporation: Field[float]       = Field(0.0,    min=0.0,    max=2.0,    color="grey",           description="Density removed per second; clears low-level haze (0 = off, 3D only)")
    dampen_threshold: Field[float]  = Field(1.2,    min=0.1,    max=5.0,    color="cyan",           description="Magnitude above which density is dampened")
    dampen_time: Field[float]       = Field(0.5,    min=0.0,    max=10.0,   color="light-blue",     description="Seconds for excess above threshold to decay to ~1%. 0=off")
    boundary: Field[BoundaryMode]   = Field(BoundaryMode.OPEN,                                      description="Open: density absorbed at edges (fluid escapes). Closed: density reflects at edges")


class TemperatureSettings(BaseSettings):
    """Temperature field parameters."""
    input_strength: Field[float]    = Field(0.0,    min=0.0,    max=2.0,    color="teal",           description="Multiplier applied to all temperature inputs")
    speed_offset: Field[float]      = Field(0.0,    min=-5.0,   max=5.0,    color="blue-grey",      description="Added to base speed for temperature only. 0 = physically coupled.")
    fade_time: Field[float]         = Field(3.0,    min=0.01,   max=60.0,   color="deep-blue",      description="Seconds until temperature fades to ~1%")
    buoyancy: Field[float]          = Field(0.0,    min=0.0,    max=10.0,   color="deep-orange",    description="Thermal buoyancy coefficient: hot air rises")
    weight: Field[float]            = Field(-10.0,  min=-20.0,  max=2.0,    color="red",            description="Ratio of gravity/settling vs thermal lift")
    ambient: Field[float]           = Field(0.2,    min=0.0,    max=1.0,    color="brown",          description="Reference temperature (buoyancy = 0 at this temp)")
    boundary: Field[BoundaryMode]   = Field(BoundaryMode.CLOSED,                                    description="Closed: insulated walls (temperature reflects). Open: cold walls (temperature=0 at edges)")


class PressureSettings(BaseSettings):
    """Pressure field parameters."""
    speed: Field[float]             = Field(0.0,    min=0.0,    max=2.0,    color="blue-grey",      description="Pressure advection speed")
    fade_time: Field[float]         = Field(8.0,    min=0.01,   max=60.0,   color="deep-blue",      description="Seconds until pressure fades to ~1%")
    iterations: Field[int]          = Field(40,     min=1,      max=60,     color="orange",         description="Solver quality for pressure (iterations at 60fps, auto-scaled for frame rate)")
    boundary: Field[BoundaryMode]   = Field(BoundaryMode.OPEN,                                      description="Open: pressure escapes at edges (reduces sticky feel at low resolution). Closed: pressure reflects")


class DepthSettings(BaseSettings):
    """Z-axis / volume parameters (ignored by 2D simulations)."""
    depth: Field[int]               = Field(4,    min=1,    max=64,     description="Density depth layers (the stored ink volume)")
    sim_depth: Field[int]           = Field(4,    min=1,    max=128,    description="Velocity/pressure/temperature depth layers (the sim grid)")
    spacing: Field[float]           = Field(4.0,  min=1.0,  max=64.0,   description="Z layer spacing in sim texels (low = lively depth, high = flat)")
    emission: Field[float]          = Field(1.0,  min=0.0,  max=8.0,    color="amber",          description="Ink brightness gain per unit density")
    gamma: Field[float]             = Field(1.0,  min=0.5,  max=3.0,    color="orange",         description="Density response exponent: >1 hides thin haze, pops dense cores")
    absorption: Field[float]        = Field(4.0,  min=0.01, max=50.0,   color="deep-orange",    description="Occlusion: extinction per unit density (near ink covers far ink)")
    glow: Field[float]              = Field(0.25, min=0.0,  max=1.0,    color="yellow",         description="Additive glow mix: 1 = legacy additive (depth invisible)")
    tint: Field[float]              = Field(0.5,  min=0.0,  max=1.0,    color="light-blue",     description="Far ink shifts toward water colour (red absorbed first)")
    fog: Field[float]               = Field(0.5,  min=0.0,  max=4.0,    color="blue-grey",      description="Water extinction over the full depth; dims far layers")
    parallax: Field[float]          = Field(0.02, min=0.0,  max=0.1,    color="purple",         description="Virtual eye offset in UV units; layers shear with depth")
    perspective: Field[float]       = Field(0.15, min=0.0,  max=1.0,    color="deep-purple",    description="Per-layer shrink toward the vanishing point; 0 = flat")
    drift_period: Field[float]      = Field(45.0, min=0.0,  max=180.0,  color="indigo",         description="Seconds per orbit of the drifting eye; 0 = static offset")
    eye_tracking: Field[float]      = Field(0.0,  min=0.0,  max=1.0,    color="violet",         description="Parallax eye follows the tracked head; 0 = autonomous drift")
    depth_blur: Field[float]        = Field(0.0,  min=0.0,  max=8.0,    color="brown",          description="Blur radius in texels at the back layer; 0 = off")
    debug_depth: Field[bool]        = Field(False,                                              description="Show dominant depth layer as hue (red = back, violet = front)")
    injection_layer: Field[float]   = Field(1.0,  min=0.0,  max=1.0,    description="Normalized depth for 2D->3D injection center")
    injection_spread: Field[float]  = Field(0.001, min=0.001, max=0.5,   description="Gaussian sigma for depth spread during injection")
    boundary: Field[BoundaryMode]   = Field(BoundaryMode.CLOSED,                                    description="Z-axis boundary (3D only). Closed: depth layers reflect at front/back. Open: depth layers absorb")


class FluidFlowSettings(BaseSettings):
    """Unified configuration for all fluid simulations (2D and 3D).

    Speed model:
        One 'speed' parameter controls all passive scalar transport (density,
        temperature, pressure). At speed=1.0, a velocity value of 1.0 moves
        fluid across the full texture width in 1 second.

    Fade_time model:
        Exponential frame-rate-independent decay: multiplier = 0.01^(dt/fade_time).
        fade_time=3.0 means the field retains ~1% after 3 seconds.

    Z-axis:
        3D simulations use the z sub-group for volume parameters.
        2D simulations ignore it.
    """

    # ---- Actions ----
    reset_sim: Field[bool] = Field(False, widget=Widget.button, description="Reset all simulation fields to zero")
    profile: Field[bool] = Field(False, description="Enable per-pass GPU timing (uses glFinish; serializes pipeline)")

    # ---- Grid dimensions (multiples of 32) ----
    width: Field[int]               = Field(1024,   min=32,     max=4096,   step=32,    description="Simulation grid width")
    height: Field[int]              = Field(576,    min=32,     max=4096,   step=32,    description="Simulation grid height")
    fps: Field[int]                 = Field(60,     min=1,      max=240,    description="Current average FPS for dt calculation (bound from WindowManager)", access=Field.READ)
    speed: Field[float]             = Field(1.0,    min=0.0,    max=5.0,    description="Base fluid transport rate")

    # ---- Field groups ----
    depth: Group[DepthSettings]             = Group(DepthSettings)
    velocity: Group[VelocitySettings]       = Group(VelocitySettings)
    density: Group[DensitySettings]         = Group(DensitySettings)
    temperature: Group[TemperatureSettings] = Group(TemperatureSettings)
    pressure: Group[PressureSettings]       = Group(PressureSettings)
