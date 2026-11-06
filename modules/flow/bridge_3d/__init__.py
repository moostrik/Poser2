"""The 3D bridge: deep_flow's velocity lowpass and bandpass, density and temperature, on a 3D velocity (xyz)."""

from .density_bridge import DensityBridge, DensityBridgeSettings
from .trails import decay_weight, direction, advect
from .velocity_lowpass import VelocityLowpass, VelocityLowpassSettings, velocity_trail, trail_acceleration
from .shaders import Change, WeightedSum
from .velocity_bandpass import VelocityBandpass, VelocityBandpassSettings, velocity_change, step_peak, moving_mean, start_region_share, stop_region_share, change_velocity, band_width, change_acceleration_scale, whole_change
from .magnitude import Magnitude, VelocityMagnitude
from .temperature_bridge import TemperatureBridge, TemperatureBridgeSettings
