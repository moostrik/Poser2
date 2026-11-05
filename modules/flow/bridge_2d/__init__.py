"""The 2D bridge, as on main: hd_trio's smooth trail, density and temperature on a 2D velocity (xy).
Kept apart from the 3D bridge so the 3D work does not change hd_trio."""

from .smooth_trail import SmoothTrail, VelocitySmoothTrail, SmoothTrailSettings
from .density_bridge import DensityBridge, DensityBridgeSettings
from .temperature_bridge import TemperatureBridge, TemperatureBridgeSettings
from .magnitude import Magnitude, VelocityMagnitude
