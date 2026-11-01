"""Bridge layers — convert raw signals into fluid-compatible fields."""

from .density_bridge import DensityBridge, DensityBridgeSettings
from .kinetic_bridge import BlurPosition, KineticBridge, KineticBridgeSettings
from .kinetic_strip import KineticStrip, LANES as KINETIC_STRIP_LANES
from .depth_velocity import DepthVelocity, DepthVelocitySettings
from .kinetics import decay_weight, project, hold_release, drop, fill_reservoir, smooth_depth, depth_speed
from .smooth_trail import SmoothTrail, VelocitySmoothTrail, DepthSmoothTrail, SmoothTrailSettings
from .magnitude import Magnitude, VelocityMagnitude
from .temperature_bridge import TemperatureBridge, TemperatureBridgeSettings
