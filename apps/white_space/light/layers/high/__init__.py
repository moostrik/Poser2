"""High-speed light layers — shown above the LOW→HIGH crossfade, with the ring light_offset applied.

Persistence-of-vision content that only reads while the bar spins fast: pose/camera-driven visuals.
"""

from .pose_waves        import PoseWaves,   PoseWavesSettings
from .harmonic          import Harmonic,    HarmonicSettings, HarmonicSourceSettings
from .player_azimuth    import PlayerLines,  PlayerLinesSettings
from .camera_light import CameraLight,  CameraLightSettings
from .playhead          import Playhead,    PlayheadSettings
