from enum import IntEnum, auto
from modules.settings import BaseSettings, Field, Group, Widget


from .clock import ClockSettings
from .motor import MotorSettings
from .playhead import PlayheadSettings
from .layers import (
    PoseWavesSettings, FillSettings, PulseSettings,
    ChaseSettings, LinesSettings, RandomSettings, HarmonicSettings,
    PlayerLinesSettings, CameraLightSettings, PlayheadFlashSettings,
    HauntedFlashSettings, PlayheadLowSettings, PlayheadHighSettings,
)


class LayerId(IntEnum):
    """The unified layer pool: everything a show state (or the manual selector) can put in a look.

    One instance per layer. The slow-speed lamp layers and the high-speed ring layers share
    the pool — a state selects what fits the motor mode it commands.
    """
    playhead_lamp   = auto()   # slow-speed front white lamp (the searchlight line)
    playhead_flash  = auto()
    haunt_flash     = auto()
    pose_waves      = auto()
    harmonic        = auto()
    player_lines    = auto()
    calibration     = auto()
    playhead_marker = auto()   # bright ring marker visualising the content playhead
    fill            = auto()
    pulse           = auto()
    chase           = auto()
    lines           = auto()
    random          = auto()


class LayerCompSettings(BaseSettings):
    """Per-layer composition settings — one group per pool layer.
    `fov` is a hidden relay (from the root) into player_lines/calibration."""
    fov: Field[float] = Field(110.0, min=60.0, max=180.0, step=0.5, visible=False, description="Camera horizontal FOV — hidden relay to player_lines/calibration")
    playhead_lamp:   Group[PlayheadLowSettings]   = Group(PlayheadLowSettings)
    playhead_flash:  Group[PlayheadFlashSettings] = Group(PlayheadFlashSettings)
    haunt_flash:     Group[HauntedFlashSettings]  = Group(HauntedFlashSettings)
    pose_waves:      Group[PoseWavesSettings]     = Group(PoseWavesSettings)
    harmonic:        Group[HarmonicSettings]      = Group(HarmonicSettings)
    player_lines:    Group[PlayerLinesSettings]   = Group(PlayerLinesSettings, share=[fov.as_('fov')])
    calibration:     Group[CameraLightSettings]   = Group(CameraLightSettings, share=[fov.as_('fov')])
    playhead_marker: Group[PlayheadHighSettings]  = Group(PlayheadHighSettings)
    fill:            Group[FillSettings]          = Group(FillSettings)
    pulse:           Group[PulseSettings]         = Group(PulseSettings)
    chase:           Group[ChaseSettings]         = Group(ChaseSettings)
    lines:           Group[LinesSettings]         = Group(LinesSettings)
    random:          Group[RandomSettings]        = Group(RandomSettings)


class LightSettings(BaseSettings):
    """Settings for the LED light system (conductor thread + layers + compositor)."""

    # Construction / wiring (INIT — requires restart to take effect)
    max_poses:        Field[int]   = Field(3,    min=1,   max=16,   access=Field.INIT, description="Max tracked poses")
    num_cameras:      Field[int]   = Field(1,    min=1,   max=16,   access=Field.INIT, description="Number of cameras")
    light_rate:       Field[float] = Field(30.0, min=1,   max=120,  access=Field.INIT, description="Light output frame rate (fps)")
    light_resolution: Field[int]   = Field(3600, min=256, max=4000, access=Field.INIT, description="LED strip resolution (pixels)")
    fov: Field[float] = Field(110.0, min=60.0, max=180.0, step=0.5, description="Camera horizontal FOV — hidden relay from root to player_lines/calibration")

    master:         Field[float] = Field(1.0, min=0.0, max=1.0, step=0.01, description="Master brightness (applied to the composite; lamp gamma/floor live in osc_light)", newline=True)
    light_phase: Field[float]    = Field(0.0, min=0.0, max=1.0, step=0.01, description="High-speed ring offset (0–1 turn), applied to the spun-content layers")

    # Debug override — a first-class switch ABOVE the state machine: while on, the
    # Compositor draws debug_layers (raw, full weight) and the motor auto-follows the
    # selection's regime (any high layer → HIGH, else any low → LOW, empty → STOPPED).
    # Forced off at startup (boot failsafe: a preset saved mid-debug must never spin at power-on).
    debug:        Field[bool]          = Field(False, description="Debug override: show debug_layers and auto-follow the motor to their regime", newline=True)
    debug_layers: Field[list[LayerId]] = Field([LayerId.playhead_lamp], widget=Widget.checklist, description="Layers shown while debug is on (full weight; motor follows their regime)")

    clock:        Group[ClockSettings]        = Group(ClockSettings)
    motor:        Group[MotorSettings]        = Group(MotorSettings)
    playhead:     Group[PlayheadSettings]     = Group(PlayheadSettings)

    # The unified per-layer composition settings pool (was low/high/test slot groups).
    layers: Group[LayerCompSettings] = Group(LayerCompSettings, share=[fov.as_('fov')])
