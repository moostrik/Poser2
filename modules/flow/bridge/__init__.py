"""Bridge layers — convert raw signals into fluid-compatible fields."""

from .density_bridge import DensityBridge, DensityBridgeSettings
from .kinetic_bridge import BlurPosition, KineticBridge, KineticBridgeSettings
from .kinetic_strip import KineticStrip, LANES as KINETIC_STRIP_LANES
from .kinetics import decay_weight, project, hold_release, drop, fill_reservoir
from .smooth_trail import SmoothTrail, VelocitySmoothTrail, SmoothTrailSettings
from .magnitude import Magnitude, VelocityMagnitude
from .temperature_bridge import TemperatureBridge, TemperatureBridgeSettings
