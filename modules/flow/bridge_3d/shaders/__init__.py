"""The 3D bridge's shaders, on a 3D velocity (xyz) and its z positions."""

from .bandpass_accelerations import BandpassAccelerations
from .bandpass_changes import BandpassChanges
from .bandpass_trails import BandpassTrails
from .lowpass_blur import LowpassBlur
from .lowpass_step import LowpassStep
from .z_position_mix import ZPositionMix
from .velocity_bandpass_region import VelocityBandpassRegion
from .velocity_bandpass_region_mean import VelocityBandpassRegionMean
from .velocity_bandpass_region_speed import VelocityBandpassRegionSpeed
from .weighted_sum import WeightedSum
