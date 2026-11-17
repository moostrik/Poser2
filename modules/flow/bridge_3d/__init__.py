"""The 3D bridge: deep_flow's velocity lowpass and bandpass, density and temperature, on a 3D velocity (xyz)."""

from .density_bridge import DensityBridge, DensityBridgeSettings
from .trails import decay_weight
from .velocity_lowpass import VelocityLowpass, VelocityLowpassSettings, gaussian_weights, blur_sigma, blur_dispatches
from .shaders import Change, WeightedSum, ZPositionMix
from .velocity_bandpass import VelocityBandpass, VelocityBandpassSettings, step_peak, band_width, change_acceleration_scale
from .magnitude import Magnitude, VelocityMagnitude
from .temperature_bridge import TemperatureBridge, TemperatureBridgeSettings
from .capture_max import CaptureMax
from .velocity_peak import VelocityPeak, VelocityPeakSettings, speed_percentile
