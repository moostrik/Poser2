"""The 3D bridge: deep_flow's velocity lowpass and bandpass, density and temperature, on a 3D velocity (xyz)."""

from .velocity_lowpass import VelocityLowpass
from .shaders import SourceMix, ZPositionMix
from .velocity_bandpass import VelocityBandpass
from .magnitude import Magnitude, VelocityMagnitude
from .bridge_3d import Bridge3D
