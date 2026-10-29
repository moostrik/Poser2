"""Bridge layers — convert raw signals into fluid-compatible fields."""

from .density_bridge import DensityBridge, DensityBridgeSettings
from .smooth_trail import SmoothTrail, VelocitySmoothTrail, SmoothTrailSettings
from .magnitude import Magnitude, VelocityMagnitude
from .temperature_bridge import TemperatureBridge, TemperatureBridgeSettings