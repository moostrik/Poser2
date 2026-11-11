"""Bridge-specific shaders."""

from .density_bridge_shader import DensityBridgeShader
from .lowpass_blur import LowpassBlur
from .trail_advect import TrailAdvect
from .velocity_bandpass_change import Change, VelocityBandpassChange
from .velocity_bandpass_direction import VelocityBandpassDirection
from .velocity_bandpass_region import VelocityBandpassRegion
from .velocity_bandpass_region_mean import VelocityBandpassRegionMean
from .velocity_bandpass_region_speed import VelocityBandpassRegionSpeed
from .velocity_trail import VelocityTrail
from .weighted_sum import WeightedSum
from .temperature_bridge_shader import TemperatureBridgeShader
