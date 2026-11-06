"""The 3D bridge: deep_flow's trail and kinetic bridges, density and temperature, on a 3D velocity (xyz)."""

from .density_bridge import DensityBridge, DensityBridgeSettings
from .kinetic_bridge import BlurPosition, KineticBridge, KineticBridgeSettings
from .kinetics import decay_weight, project, stop_gate, hold_release, drop, fill_reservoir, advect
from .velocity_trail_bridge import VelocityTrailBridge, VelocityTrailBridgeSettings, velocity_trail
from .shaders import MotionEvent
from .velocity_event_bridge import VelocityEventBridge, VelocityEventBridgeSettings, velocity_event, step_peak, moving_mean, start_region_share, stop_region_share, event_velocity, START_FLOOR
from .magnitude import Magnitude, VelocityMagnitude
from .temperature_bridge import TemperatureBridge, TemperatureBridgeSettings
