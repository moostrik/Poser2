"""Projection-mode light layers (`ProjectionLayer`) — persistence-of-vision content that only
reads while the bar spins fast. Each authors in azimuth and nothing rotates it here; the light
sender applies the projection offset on the way out. Show layers and ``test_``-prefixed debug
layers live side by side — the name carries the role, the folder carries the mode. A class is
named after its file (``test_fill.py`` → ``TestFill``), which is also its ``LayerId``.
"""

from .pose_instrument     import PoseInstrument,     PoseInstrumentSettings, Pattern, Oscillator
from .line_pattern        import LinePattern,        Waveform
from .projection_playhead import ProjectionPlayhead, ProjectionPlayheadSettings
from .flood               import Flood,              FloodSettings
from .test_pose_waves     import TestPoseWaves,      TestPoseWavesSettings
from .test_harmonic       import TestHarmonic,       TestHarmonicSettings, TestHarmonicSourceSettings
from .test_player_lines   import TestPlayerLines,    TestPlayerLinesSettings
from .test_calibration    import TestCalibration,    TestCalibrationSettings
from .test_fill           import TestFill,           TestFillSettings
from .test_pulse          import TestPulse,          TestPulseSettings
from .test_chase          import TestChase,          TestChaseSettings
from .test_lines          import TestLines,          TestLinesSettings
from .test_random         import TestRandom,         TestRandomSettings, TestRandomChannelSettings
