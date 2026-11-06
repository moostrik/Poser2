"""Bridge-specific shaders."""

from .density_bridge_shader import DensityBridgeShader
from .trail_advect import TrailAdvect
from .velocity_impulse import Impulse, VelocityImpulse
from .velocity_impulse_direction import VelocityImpulseDirection
from .velocity_impulse_region import VelocityImpulseRegion
from .velocity_impulse_region_mean import VelocityImpulseRegionMean
from .velocity_impulse_region_speed import VelocityImpulseRegionSpeed
from .velocity_trail import VelocityTrail
from .weighted_sum import WeightedSum
from .temperature_bridge_shader import TemperatureBridgeShader
