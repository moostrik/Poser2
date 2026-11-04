"""Bridge layers — convert raw signals into fluid-compatible fields: the 2D bridge (hd_trio) and the 3D bridge (deep_flow)."""

from . import bridge_2d
from .bridge_3d import DensityBridge, DensityBridgeSettings, BlurPosition, KineticBridge, KineticBridgeSettings, \
                       KineticStrip, KINETIC_STRIP_LANES, decay_weight, project, stop_gate, hold_release, drop, fill_reservoir, advect, \
                       VelocityTrailBridge, VelocityTrailBridgeSettings, shape_velocity, velocity_trail, Magnitude, VelocityMagnitude, TemperatureBridge, TemperatureBridgeSettings
