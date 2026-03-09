"""Shared configuration sub-groups for 2D and 3D fluid simulations.

Groups fields by physical domain (velocity, density, temperature, pressure)
with prefixes stripped — the group name provides context.
"""

from modules.settings import Field, Settings, Widget


class VelocityConfig(Settings):
    """Velocity field parameters."""
    input_strength: Field[float]    = Field(1.0,    min=0.0,    max=10.0,   color="teal",           description="Multiplier applied to all velocity inputs")
    self_advection: Field[float]    = Field(0.01,   min=0.0,    max=0.2,    color="blue-grey",      description="How much velocity advects itself. Keep low for stability.")
    fade_time: Field[float]         = Field(30.0,   min=0.1,    max=60.0,   color="deep-blue",      description="Seconds until velocity fades to ~1%")
    dampen_threshold: Field[float]  = Field(5.0,    min=0.1,    max=50.0,   color="cyan",           description="Magnitude above which velocity is dampened")
    dampen_time: Field[float]       = Field(0.5,    min=0.0,    max=10.0,   color="light-blue",     description="Seconds for excess above threshold to decay to ~1%. 0=off")
    vorticity: Field[float]         = Field(5.17,   min=0.0,    max=60.0,   color="purple",         description="Vortex confinement strength (adds turbulence)")
    vorticity_radius: Field[float]  = Field(2.82,   min=1.0,    max=30.0,   color="deep-purple",    description="Curl sampling radius in texels")
    viscosity: Field[float]         = Field(6.62,   min=0.0,    max=100.0,  color="indigo",         description="Fluid thickness/resistance to flow")
    viscosity_iter: Field[int]      = Field(40,     min=1,      max=60,     color="orange",         description="Solver quality for viscosity (iterations at 60fps, auto-scaled for frame rate)")


class DensityConfig(Settings):
    """Density field parameters."""
    input_strength: Field[float]    = Field(1.0,    min=0.0,    max=2.0,    color="teal",           description="Multiplier applied to all density inputs")
    speed_offset: Field[float]      = Field(0.0,    min=-5.0,   max=5.0,    color="blue-grey",      description="Added to base speed for density only. 0 = physically coupled.")
    fade_time: Field[float]         = Field(4.0,    min=0.01,   max=60.0,   color="deep-blue",      description="Seconds until density fades to ~1%")
    dampen_threshold: Field[float]  = Field(1.2,    min=0.1,    max=5.0,    color="cyan",           description="Magnitude above which density is dampened")
    dampen_time: Field[float]       = Field(0.5,    min=0.0,    max=10.0,   color="light-blue",     description="Seconds for excess above threshold to decay to ~1%. 0=off")


class TemperatureConfig(Settings):
    """Temperature field parameters."""
    input_strength: Field[float]    = Field(0.0,    min=0.0,    max=2.0,    color="teal",           description="Multiplier applied to all temperature inputs")
    speed_offset: Field[float]      = Field(0.0,    min=-5.0,   max=5.0,    color="blue-grey",      description="Added to base speed for temperature only. 0 = physically coupled.")
    fade_time: Field[float]         = Field(3.0,    min=0.01,   max=60.0,   color="deep-blue",      description="Seconds until temperature fades to ~1%")
    buoyancy: Field[float]          = Field(0.0,    min=0.0,    max=10.0,   color="deep-orange",    description="Thermal buoyancy coefficient: hot air rises")
    weight: Field[float]            = Field(-10.0,  min=-20.0,  max=2.0,    color="red",            description="Ratio of gravity/settling vs thermal lift")
    ambient: Field[float]           = Field(0.2,    min=0.0,    max=1.0,    color="brown",          description="Reference temperature (buoyancy = 0 at this temp)")


class PressureConfig(Settings):
    """Pressure field parameters."""
    speed: Field[float]             = Field(0.0,    min=0.0,    max=2.0,    color="blue-grey",      description="Pressure advection speed")
    fade_time: Field[float]         = Field(8.0,    min=0.01,   max=60.0,   color="deep-blue",      description="Seconds until pressure fades to ~1%")
    iterations: Field[int]          = Field(40,     min=1,      max=60,     color="orange",         description="Solver quality for pressure (iterations at 60fps, auto-scaled for frame rate)")


class DepthConfig(Settings):
    """Z-axis / volume parameters (ignored by 2D simulations)."""
    depth: Field[int]               = Field(4,    min=1,    max=64,     description="Number of depth layers in the 3D volume")
    scale: Field[float]             = Field(1.0,  min=0.5,  max=5.0,    description="Manual multiplier on auto-computed Z grid spacing (width/depth)")
    composite_mode: Field[int]      = Field(3,    min=0,    max=4,      description="3D->2D compositing: 0=alpha, 1=additive, 2=max, 3=emission-absorption, 4=debug depth")
    ray_steps: Field[int]           = Field(32,   min=1,    max=128,    description="Number of ray-march steps for volumetric composite (mode 3). More steps = smoother inter-layer interpolation")
    absorption: Field[float]        = Field(4.0,  min=0.01,  max=50.0,   description="Beer's law absorption coefficient (higher = more opaque per unit density)")
    injection_layer: Field[float]   = Field(1.0,  min=0.0,  max=1.0,    description="Normalized depth for 2D->3D injection center")
    injection_spread: Field[float]  = Field(0.001, min=0.001, max=0.5,   description="Gaussian sigma for depth spread during injection")


class FluidFlowConfig(Settings):
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

    # ---- Grid dimensions (multiples of 32) ----
    width: Field[int]               = Field(1024,   min=32,     max=4096,   step=32,    description="Simulation grid width")
    height: Field[int]              = Field(576,    min=32,     max=4096,   step=32,    description="Simulation grid height")
    fps: Field[int]                 = Field(60,     min=1,      max=240,    description="Current average FPS for dt calculation (bound from WindowManager)", access=Field.READ)
    speed: Field[float]             = Field(1.0,    min=0.0,    max=5.0,    description="Base fluid transport rate")

    # ---- Field groups ----
    depth:       DepthConfig
    velocity:    VelocityConfig
    density:     DensityConfig
    temperature: TemperatureConfig
    pressure:    PressureConfig
