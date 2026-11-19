"""The bridges' settings, the 2D's and the 3D's: the filters that make the readings, then one group per output, each
with its mix."""

from modules.settings import BaseSettings, Field, Group, Widget
from .velocity_peak import VelocityPeakSettings


class VelocityLowpassSettings(BaseSettings):
    """Configuration for the velocity lowpass."""
    advect = Field(0.0, min=0.0, max=1.0, step=0.05, description="How far the trail moves with its motion before each step; 0 = at a fixed pixel")
    attack = Field(0.0, min=0.0, max=500.0, step=1.0, description="Smoothing time of a rising speed, ms; 0 = as smooth (lower: quicker flicks)")
    smooth = Field(160.0, min=0.0, max=3000.0, step=1.0, description="Smoothing time of a falling speed, ms; 0 = none (higher: longer coast)")
    blur = Field(0.0, min=0.0, max=0.3, step=0.001, newline=True, description="How far the trail spreads in 1 s, widths at half its height; 0 = off")


class VelocityBandpassSettings(BaseSettings):
    """Configuration for the velocity bandpass: its two trails and the start and stop regions."""
    fast_smooth = Field(25.0, min=0.0, max=500.0, step=1.0, description="Fast trail smoothing time, ms; 0 = the flow itself")
    slow_smooth = Field(150.0, min=0.0, max=1000.0, step=1.0, description="Slow trail smoothing time, ms: what a change is read against")
    start_region_weight = Field(1.0, min=0.0, max=1.0, step=0.05, newline=True, description="How much the region around a start must speed up too, the rest the bow; 0 = all start")
    stop_region_weight = Field(1.0, min=0.0, max=1.0, step=0.05, description="How much the region around a stop must slow down too, the rest the wake; 0 = all stop")
    start_region_size = Field(128.0, min=8.0, max=1024.0, step=8.0, newline=True, description="Size of the region around a start, pixels of the bridge")
    stop_region_size = Field(128.0, min=8.0, max=1024.0, step=8.0, description="Size of the region around a stop, pixels of the bridge")


class MixSettings(BaseSettings):
    """What an output is made of: each reading at its share, 0 to 1, as knobs on the first row; none by default."""
    trail = Field(0.0, min=0.0, max=1.0, step=0.05, widget=Widget.knob, description="Share of the trail: the motion, held and smoothed")
    change = Field(0.0, min=0.0, max=1.0, step=0.05, widget=Widget.knob, description="Share of the change: the whole change of speed, signed")
    start = Field(0.0, min=0.0, max=1.0, step=0.05, widget=Widget.knob, description="Share of the start: a part of the body speeding up")
    bow = Field(0.0, min=0.0, max=1.0, step=0.05, widget=Widget.knob, description="Share of the bow: a limb arriving")
    stop = Field(0.0, min=0.0, max=1.0, step=0.05, widget=Widget.knob, description="Share of the stop: a part of the body slowing down")
    wake = Field(0.0, min=0.0, max=1.0, step=0.05, widget=Widget.knob, description="Share of the wake: a limb leaving")


class TrailMixSettings(MixSettings):
    """A mix of the trail by default: a boundary output's."""
    trail = Field(1.0, min=0.0, max=1.0, step=0.05, widget=Widget.knob, description="Share of the trail: the motion, held and smoothed")


class VelocityMixSettings(TrailMixSettings):
    """The velocity, the boundary the water is dragged toward: its mix, and the peak its speed reached."""
    peak = Group(VelocityPeakSettings)


class AmountMixSettings(MixSettings):
    """An amount's mix: the readings' speeds and the mask, a presence while still."""
    mask = Field(0.0, min=0.0, max=1.0, step=0.05, widget=Widget.knob, description="Share of the mask: the player there, still or not")


class TrailAmountMixSettings(AmountMixSettings, TrailMixSettings):
    """An amount's mix of the trail by default: a boundary amount's."""


class DensityMixSettings(TrailAmountMixSettings):
    """The density, held at the body as a boundary value: its amount's mix; its color is the image's."""


class TemperatureMixSettings(TrailAmountMixSettings):
    """The temperature, held at the body as a boundary value: its amount's mix, a presence capped at 1; the
    fluid's `temperature_gain` makes it the warmth."""


class BridgeFiltersSettings(BaseSettings):
    """The bridge's filters, which make the readings from the motion: the lowpass and the bandpass."""
    lowpass  = Group(VelocityLowpassSettings)
    bandpass = Group(VelocityBandpassSettings)


class ExpansionSettings(AmountMixSettings):
    """The expansion: where the water swells, a presence 0 to 1 from the readings' speeds and the mask."""


class BridgeSettings(BaseSettings):
    """Configuration for a bridge, the 2D's and the 3D's: the filters, then one group per output, named as the
    output, in the order the bridge makes them: the velocity, the boundary the water is dragged toward (the
    readings mixed as velocities), the acceleration, the body force added × dt (the readings mixed as
    accelerations), the density and the temperature, held at the body as boundary values, and the expansion."""
    filters      = Group(BridgeFiltersSettings)
    velocity     = Group(VelocityMixSettings)
    acceleration = Group(MixSettings)
    density      = Group(DensityMixSettings)
    temperature  = Group(TemperatureMixSettings)
    expansion    = Group(ExpansionSettings)
