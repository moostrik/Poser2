"""Bridge layers — convert raw signals into fluid-compatible fields."""

from .DensityBridge import DensityBridge, DensityBridgeSettings
from .SmoothTrail import SmoothTrail, VelocitySmoothTrail, SmoothTrailSettings
from .Magnitude import Magnitude, VelocityMagnitude
from .TemperatureBridge import TemperatureBridge, TemperatureBridgeSettings