"""The 3D bridge: deep_flow's velocity trail and impulse bridges, density and temperature, on a 3D velocity (xyz)."""

from .density_bridge import DensityBridge, DensityBridgeSettings
from .trails import decay_weight, direction, advect
from .velocity_trail_bridge import VelocityTrailBridge, VelocityTrailBridgeSettings, velocity_trail
from .shaders import Impulse, WeightedSum
from .velocity_impulse_bridge import VelocityImpulseBridge, VelocityImpulseBridgeSettings, velocity_impulse, step_peak, moving_mean, start_region_share, stop_region_share, impulse_velocity
from .magnitude import Magnitude, VelocityMagnitude
from .temperature_bridge import TemperatureBridge, TemperatureBridgeSettings
