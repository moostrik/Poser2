"""The 3D bridge: deep_flow's velocity trail and change bridges, density and temperature, on a 3D velocity (xyz)."""

from .density_bridge import DensityBridge, DensityBridgeSettings
from .trails import decay_weight, direction, advect
from .velocity_trail_bridge import VelocityTrailBridge, VelocityTrailBridgeSettings, velocity_trail
from .shaders import Change, WeightedSum
from .velocity_change_bridge import VelocityChangeBridge, VelocityChangeBridgeSettings, velocity_change, step_peak, moving_mean, start_region_share, stop_region_share, change_velocity
from .magnitude import Magnitude, VelocityMagnitude
from .temperature_bridge import TemperatureBridge, TemperatureBridgeSettings
