"""High-regime light layers (`HighLayer`) — persistence-of-vision ring content that only
reads while the bar spins fast; all carry the ``light_phase`` ring shift. Show layers and
``test_``-prefixed debug layers live side by side — the name carries the role, the folder
carries the regime.
"""

from .pose_waves        import PoseWaves,   PoseWavesSettings
from .harmonic          import Harmonic,    HarmonicSettings, HarmonicSourceSettings
from .player_azimuth    import PlayerLines,  PlayerLinesSettings
from .camera_light      import CameraLight,  CameraLightSettings
from .playhead          import Playhead,    PlayheadSettings
from .fill              import Fill,   FillSettings
from .pulse             import Pulse,  PulseSettings
from .chase             import Chase,  ChaseSettings
from .lines             import Lines,  LinesSettings
from .random            import Random, RandomSettings, RandomChannelSettings
