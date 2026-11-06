"""Bridge-specific shaders."""

from .density_bridge_shader import DensityBridgeShader
from .trail_advect import TrailAdvect
from .velocity_change import Change, VelocityChange
from .velocity_change_direction import VelocityChangeDirection
from .velocity_change_region import VelocityChangeRegion
from .velocity_change_region_mean import VelocityChangeRegionMean
from .velocity_change_region_speed import VelocityChangeRegionSpeed
from .velocity_trail import VelocityTrail
from .weighted_sum import WeightedSum
from .temperature_bridge_shader import TemperatureBridgeShader
