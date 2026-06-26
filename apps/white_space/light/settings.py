from enum import IntEnum, auto
from modules.settings import BaseSettings, Field, Group
from modules.settings.widget import Widget


from .clock import ClockSettings
from .motor import MotorSettings
from .playhead import PlayheadSettings
from .layers import (
    PoseWavesSettings, FillSettings, PulseSettings,
    ChaseSettings, LinesSettings, RandomSettings, HarmonicSettings,
    PlayerLinesSettings, CalibrationSettings, PlayheadFlashSettings,
)


class LevelsSettings(BaseSettings):
    master:     Field[float] = Field(1.0,  min=0.0, max=1.0, step=0.01, description="Master brightness")
    lower_edge: Field[float] = Field(0.35, min=0.0, max=1.0, step=0.01, description="Lamp turn-on floor: lit pixels lift to at least this; black stays off")
    curve:      Field[float] = Field(1.0,  min=0.5, max=2.0, step=0.01, description="Brightness power curve (gamma); <1 brightens mids, >1 darkens")


class LayerId(IntEnum):
    pose_waves   = auto()
    fill         = auto()
    pulse        = auto()
    chase        = auto()
    lines        = auto()
    random       = auto()
    harmonic     = auto()
    player_lines   = auto()
    calibration    = auto()
    playhead_flash = auto()


class LightSettings(BaseSettings):
    """Settings for the LED light renderer thread."""

    # Construction / wiring (INIT — requires restart to take effect)
    max_poses:        Field[int]   = Field(3,    min=1,   max=16,   access=Field.INIT, description="Max tracked poses")
    num_cameras:      Field[int]   = Field(1,    min=1,   max=16,   access=Field.INIT, visible=False, description="Number of cameras")
    light_rate:       Field[float] = Field(30.0, min=1,   max=120,  access=Field.INIT, description="Light output frame rate (fps)")
    light_resolution: Field[int]   = Field(3600, min=256, max=4000, access=Field.INIT, visible=False, description="LED strip resolution (pixels)")
    fov: Field[float] = Field(110.0, min=60.0, max=180.0, step=0.5, visible=False, description="Camera horizontal FOV — hidden relay from root to player_lines/calibration")

    active: Field[list[LayerId]] = Field([LayerId.pose_waves], widget=Widget.checklist,description="Active layers", newline=True)

    master:    Field[float] = Field(1.0, min=0.0, max=1.0, step=0.01, description="Master brightness", newline=True)

    clock:        Group[ClockSettings]        = Group(ClockSettings)
    motor:        Group[MotorSettings]        = Group(MotorSettings)
    playhead:     Group[PlayheadSettings]     = Group(PlayheadSettings)
    levels:       Group[LevelsSettings]       = Group(LevelsSettings, share=[master.as_('master')])
    pose_waves:   Group[PoseWavesSettings]    = Group(PoseWavesSettings)
    fill:         Group[FillSettings]         = Group(FillSettings)
    pulse:        Group[PulseSettings]        = Group(PulseSettings)
    chase:        Group[ChaseSettings]        = Group(ChaseSettings)
    lines:        Group[LinesSettings]        = Group(LinesSettings)
    random:       Group[RandomSettings]       = Group(RandomSettings)
    harmonic:     Group[HarmonicSettings]     = Group(HarmonicSettings)
    player_lines:   Group[PlayerLinesSettings]    = Group(PlayerLinesSettings,  share=[fov.as_('fov')])
    calibration:    Group[CalibrationSettings]     = Group(CalibrationSettings,   share=[fov.as_('fov')])
    playhead_flash: Group[PlayheadFlashSettings]   = Group(PlayheadFlashSettings)
