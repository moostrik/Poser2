"""Shared configuration sub-groups for 2D and 3D fluid simulations.

Groups fields by physical domain (velocity, density, temperature, pressure)
with prefixes stripped — the group name provides context.
"""

from modules.settings import Field, Settings, Widget


class VelocityConfig(Settings):
    """Velocity field parameters."""
    input_strength: Field[float]    = Field(1.0,  min=0.0, max=10.0,  color="primary", description="Multiplier applied to all velocity inputs")
    self_advection: Field[float]    = Field(0.01, min=0.0, max=0.2,   description="How much velocity advects itself. Keep low for stability.")
    lifetime: Field[float]          = Field(30.0, min=0.1, max=60.0,  description="Seconds until velocity fades to ~1%")
    clamp_max: Field[float]         = Field(5.0,  min=0.1, max=50.0,  description="Maximum velocity magnitude after clamping")
    vorticity: Field[float]         = Field(5.17, min=0.0, max=60.0,  description="Vortex confinement strength (adds turbulence)")
    vorticity_radius: Field[float]  = Field(2.82, min=1.0, max=30.0,  description="Curl sampling radius in texels")
    viscosity: Field[float]         = Field(6.62, min=0.0, max=100.0, description="Fluid thickness/resistance to flow")
    viscosity_iter: Field[int]      = Field(40,   min=1,   max=60,    description="Solver iterations for viscosity")


class DensityConfig(Settings):
    """Density field parameters."""
    input_strength: Field[float]    = Field(0.01, min=0.0, max=1.0,   description="Multiplier applied to all density inputs")
    speed_offset: Field[float]      = Field(0.0,  min=-5.0, max=5.0,  description="Added to base speed for density only. 0 = physically coupled.")
    lifetime: Field[float]          = Field(4.0,  min=0.01, max=60.0,  description="Seconds until density fades to ~1%")
    clamp_max: Field[float]         = Field(1.2,  min=0.1, max=5.0,   description="Maximum density value after clamping")


class TemperatureConfig(Settings):
    """Temperature field parameters."""
    input_strength: Field[float]    = Field(0.0,   min=0.0, max=1.0,   description="Multiplier applied to all temperature inputs")
    speed_offset: Field[float]      = Field(0.0,   min=-5.0, max=5.0,  description="Added to base speed for temperature only. 0 = physically coupled.")
    lifetime: Field[float]          = Field(3.0,   min=0.01, max=60.0,  description="Seconds until temperature fades to ~1%")
    buoyancy: Field[float]          = Field(0.0,   min=0.0, max=10.0,   description="Thermal buoyancy coefficient: hot air rises")
    weight: Field[float]            = Field(-10.0, min=-20.0, max=2.0,  description="Ratio of gravity/settling vs thermal lift")
    ambient: Field[float]           = Field(0.2,   min=0.0, max=1.0,    description="Reference temperature (buoyancy = 0 at this temp)")


class PressureConfig(Settings):
    """Pressure field parameters."""
    speed: Field[float]             = Field(0.0,  min=0.0, max=2.0,      description="Pressure advection speed")
    lifetime: Field[float]          = Field(8.0,  min=0.01, max=60.0,  description="Seconds until pressure fades to ~1%")
    iterations: Field[int]          = Field(40,   min=1,   max=60,     description="Solver iterations for pressure")
