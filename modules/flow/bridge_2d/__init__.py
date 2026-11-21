"""The 2D bridge: the velocity lowpass and bandpass and the outputs' mixes on a 2D velocity (xy), each its own 2D
kernel."""

from .shaders import SpeedCount2D, SourceMix2D
from .velocity_lowpass_2d import VelocityLowpass2D
from .velocity_bandpass_2d import VelocityBandpass2D
from .bridge_2d import Bridge2D
