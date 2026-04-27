from modules.settings import BaseSettings, Field, Group

from .transport import TransportSettings
from .comps import (
    PoseWavesSettings, FillSettings, PulseSettings,
    ChaseSettings, LinesSettings, RandomSettings, HarmonicSettings,
)


class CompositorSettings(BaseSettings):
    """Settings for the LED composition thread."""

    # Construction / wiring (INIT — requires restart to take effect)
    max_poses:        Field[int]   = Field(3,    min=1,   max=16,   access=Field.INIT, description="Max tracked poses")
    light_rate:       Field[float] = Field(30.0, min=1,   max=120,  access=Field.INIT, description="Light output frame rate (fps)")
    light_resolution: Field[int]   = Field(3600, min=256, max=4000, access=Field.INIT, visible=False, description="LED strip resolution (pixels)")

    transport:  Group[TransportSettings]  = Group(TransportSettings)
    pose_waves: Group[PoseWavesSettings]  = Group(PoseWavesSettings)
    fill:       Group[FillSettings]       = Group(FillSettings)
    pulse:      Group[PulseSettings]      = Group(PulseSettings)
    chase:      Group[ChaseSettings]      = Group(ChaseSettings)
    lines:      Group[LinesSettings]      = Group(LinesSettings)
    random:     Group[RandomSettings]     = Group(RandomSettings)
    harmonic:   Group[HarmonicSettings]   = Group(HarmonicSettings)
