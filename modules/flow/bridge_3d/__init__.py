"""The 3D bridge: deep_flow's velocity trail and event bridges, density and temperature, on a 3D velocity (xyz)."""

from .density_bridge import DensityBridge, DensityBridgeSettings
from .trails import decay_weight, direction, advect
from .velocity_trail_bridge import VelocityTrailBridge, VelocityTrailBridgeSettings, velocity_trail
from .shaders import MotionEvent
from .velocity_event_bridge import VelocityEventBridge, VelocityEventBridgeSettings, velocity_event, step_peak, moving_mean, start_region_share, stop_region_share, event_velocity
from .magnitude import Magnitude, VelocityMagnitude
from .temperature_bridge import TemperatureBridge, TemperatureBridgeSettings
