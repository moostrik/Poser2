from enum import IntEnum, auto
from modules.settings import BaseSettings, Field, Group
from modules.settings.widget import Widget

from .draw import BlendType
from .comps import (
    PoseWavesSettings, FillSettings, PulseSettings,
    ChaseSettings, LinesSettings, RandomSettings, HarmonicSettings,
)


class CompositionId(IntEnum):
    pose_waves = auto()
    fill       = auto()
    pulse      = auto()
    chase      = auto()
    lines      = auto()
    random     = auto()
    harmonic   = auto()


class CompositorSettings(BaseSettings):
    """Settings for the LED composition thread."""

    # Construction / wiring (INIT — requires restart to take effect)
    max_poses:        Field[int]   = Field(3,    min=1,   max=16,   access=Field.INIT, description="Max tracked poses")
    light_rate:       Field[float] = Field(30.0, min=1,   max=120,  access=Field.INIT, description="Light output frame rate (fps)")
    light_resolution: Field[int]   = Field(3600, min=256, max=4000, access=Field.INIT, visible=False, description="LED strip resolution (pixels)")

    bpm:   Field[float] = Field(120.0, min=20.0, max=480.0, step=0.5,  description="Master tempo (BPM)", newline=True)
    time:  Field[float] = Field(0.0,  access=Field.READ,               description="Elapsed wall-clock time (s)")
    phase: Field[float] = Field(0.0,  access=Field.READ,               description="Beat phase (0–1)")

    active: Field[list[CompositionId]] = Field([CompositionId.pose_waves], widget=Widget.checklist,description="Active compositions", newline=True)
    blend: Field[BlendType] = Field(BlendType.ADD, description="Blend mode")

    master:    Field[float] = Field(1.0, min=0.0, max=1.0, step=0.01, description="Master brightness", newline=True)
    hardness:  Field[float] = Field(0.0, min=0.0, max=1.0, step=0.01, description="Contrast hardness (0=off, 1=hard step)")
    threshold: Field[float] = Field(0.5, min=0.0, max=1.0, step=0.01, description="Hardness pivot point")

    pose_waves: Group[PoseWavesSettings]  = Group(PoseWavesSettings)
    fill:       Group[FillSettings]       = Group(FillSettings)
    pulse:      Group[PulseSettings]      = Group(PulseSettings)
    chase:      Group[ChaseSettings]      = Group(ChaseSettings)
    lines:      Group[LinesSettings]      = Group(LinesSettings)
    random:     Group[RandomSettings]     = Group(RandomSettings)
    harmonic:   Group[HarmonicSettings]   = Group(HarmonicSettings)
