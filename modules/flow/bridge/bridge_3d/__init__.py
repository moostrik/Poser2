"""The 3D bridge: deep_flow's trail and kinetic bridges, density and temperature, on a 3D velocity (xyz)."""

from .density_bridge import DensityBridge, DensityBridgeSettings
from .kinetic_bridge import BlurPosition, KineticBridge, KineticBridgeSettings
from .kinetic_strip import KineticStrip, LANES as KINETIC_STRIP_LANES
from .kinetics import decay_weight, project, stop_gate, hold_release, drop, fill_reservoir, advect
from .velocity_trail_bridge import VelocityTrailBridge, VelocityTrailBridgeSettings, Smoothing, velocity_trail, one_euro_rate, one_euro_trail
from .magnitude import Magnitude, VelocityMagnitude
from .temperature_bridge import TemperatureBridge, TemperatureBridgeSettings
