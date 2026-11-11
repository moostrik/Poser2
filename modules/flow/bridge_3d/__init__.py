"""The 3D bridge: deep_flow's velocity lowpass and bandpass, density and temperature, on a 3D velocity (xyz)."""

from .density_bridge import DensityBridge, DensityBridgeSettings
from .trails import decay_weight, direction, advect
from .velocity_lowpass import VelocityLowpass, VelocityLowpassSettings, velocity_trail, trail_acceleration, z_position_trail, gaussian_blur, gaussian_weights, blur_sigma, blur_dispatches
from .z_position_mix import z_position_mix
from .shaders import Change, WeightedSum, ZPositionMix
from .velocity_bandpass import VelocityBandpass, VelocityBandpassSettings, velocity_change, step_peak, moving_mean, start_region_share, stop_region_share, change_velocity, band_width, change_acceleration_scale, whole_change
from .magnitude import Magnitude, VelocityMagnitude
from .temperature_bridge import TemperatureBridge, TemperatureBridgeSettings
