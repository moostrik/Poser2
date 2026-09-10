"""Projection-mode light layers (`ProjectionLayer`) — persistence-of-vision ring content that
only reads while the bar spins fast. Each authors in azimuth and nothing rotates it here; the
light sender applies the projection offset on the way out. Show layers and ``test_``-prefixed
debug layers live side by side — the name carries the role, the folder carries the mode.
"""

from .pose_instrument     import PoseInstrument,     PoseInstrumentSettings
from .projection_playhead import ProjectionPlayhead, ProjectionPlayheadSettings
from .flood               import Flood,          FloodSettings
from .test_pose_waves     import PoseWaves,      PoseWavesSettings
from .test_harmonic       import Harmonic,       HarmonicSettings, HarmonicSourceSettings
from .test_player_lines   import PlayerLines,    PlayerLinesSettings
from .test_calibration    import CameraLight,    CameraLightSettings
from .test_fill           import Fill,           FillSettings
from .test_pulse          import Pulse,          PulseSettings
from .test_chase          import Chase,          ChaseSettings
from .test_lines          import Lines,          LinesSettings
from .test_random         import Random,         RandomSettings, RandomChannelSettings
