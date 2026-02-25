"""Bridge layers — convert raw signals into fluid-compatible fields."""

from .DensityBridge import DensityBridge, DensityBridgeConfig
from .SmoothTrail import SmoothTrail, VelocitySmoothTrail, SmoothTrailConfig
from .Magnitude import Magnitude, VelocityMagnitude
from .TemperatureBridge import TemperatureBridge, TemperatureBridgeConfig