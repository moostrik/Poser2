"""Test light layers — generic patterns usable by both the low and high slots.

Simple, content-independent waveforms (fill, pulse, chase, lines, random) handy as test/fallback
compositions at any motor speed.
"""

from .fill   import Fill,   FillSettings
from .pulse  import Pulse,  PulseSettings
from .chase  import Chase,  ChaseSettings
from .lines  import Lines,  LinesSettings
from .random import Random, RandomSettings, RandomChannelSettings
