"""The 3D bridge's settings: the readings, then one group per output, each with its mix."""

from modules.settings import BaseSettings, Field, Group, Widget
from .velocity_bandpass import VelocityBandpassSettings
from .velocity_lowpass import VelocityLowpassSettings
from .velocity_peak import VelocityPeakSettings


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


class Bridge3DFiltersSettings(BaseSettings):
    """The bridge's filters, which make the readings from the motion: the lowpass and the bandpass."""
    lowpass  = Group(VelocityLowpassSettings)
    bandpass = Group(VelocityBandpassSettings)


class ExpansionSettings(AmountMixSettings):
    """The expansion: where the water swells, a presence 0 to 1 from the readings' speeds and the mask."""


class Bridge3DSettings(BaseSettings):
    """Configuration for Bridge3D: the filters, then one group per output, named as the output, in the order the
    bridge makes them: the velocity, the boundary the water is dragged toward (the readings mixed as velocities),
    the acceleration, the body force added × dt (the readings mixed as accelerations), the density and the
    temperature, held at the body as boundary values, and the expansion."""
    filters      = Group(Bridge3DFiltersSettings)
    velocity     = Group(VelocityMixSettings)
    acceleration = Group(MixSettings)
    density      = Group(DensityMixSettings)
    temperature  = Group(TemperatureMixSettings)
    expansion    = Group(ExpansionSettings)
