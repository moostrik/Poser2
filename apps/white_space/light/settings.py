from enum import IntEnum, auto
from modules.settings import BaseSettings, Field, Group


from .clock import ClockSettings
from .motor import MotorSettings
from .playhead import PlayheadSettings
from .layers import (
    PoseWavesSettings, FillSettings, PulseSettings,
    ChaseSettings, LinesSettings, RandomSettings, HarmonicSettings,
    PlayerLinesSettings, CameraLightSettings, PlayheadFlashSettings,
    PlayheadHauntedSettings, PlayheadLowSettings, PlayheadHighSettings,
    PlayheadTestSettings, SoundLightSettings, PoseInstrumentSettings, FloodSettings,
    WindDownSettings,
)


class LayerId(IntEnum):
    """The unified layer pool: everything a state (or the debug select) can put in a mix.

    One instance per layer; each layer's regime (low = lamps, high = ring) lives in its
    class (`LowLayer`/`HighLayer`). Member order is the low block then the high block —
    it drives the `DebugLayer` dropdown order and mirrors the low/high settings groups.
    Debug-only layers (never in a state's mix) state that role in their docstrings; the
    high block's generic patterns keep the ``test_`` prefix.
    """
    # low — the lamp regime
    sound_light         = auto()   # low: soundscape levels on the left/right blue lamps
    playhead_low        = auto()   # low: front white lamp (the searchlight line)
    playhead_flash      = auto()   # low: flash as the playhead crosses a participant
    playhead_haunted    = auto()   # low: player/ghost flash (debug/experimentation)
    playhead_test       = auto()   # low: direct levels for the four physical lamps (debug)
    # high — the POV ring regime
    pose_instrument     = auto()   # high: the pose instrument (placeholder — see LAYERS.md)
    playhead_high       = auto()   # high: bright ring marker visualising the content playhead
    flood               = auto()   # high: constant full-strip white (S7's wall)
    wind_down           = auto()   # cross-regime (classed LowLayer) — flood's ending: fades ring + lamps, finishing one bar after motor lock
    test_pose_waves     = auto()   # high: the old wave/void instrument (reference/montage)
    test_harmonic       = auto()
    test_player_lines   = auto()
    test_calibration    = auto()
    test_fill           = auto()
    test_pulse          = auto()
    test_chase          = auto()
    test_lines          = auto()
    test_random         = auto()


# The debug select: OFF = debug disarmed (the show runs); any other member = debug on,
# showing exactly that layer solo while the motor auto-follows its regime. Generated from
# LayerId so it can never drift from the pool.
DebugLayer = IntEnum('DebugLayer', {'OFF': 0, **{m.name: m.value for m in LayerId}})


class LowLayersSettings(BaseSettings):
    """Per-layer composition settings — the low (lamp-regime) block of the pool."""
    sound_light:        Group[SoundLightSettings]       = Group(SoundLightSettings)
    playhead_low:       Group[PlayheadLowSettings]      = Group(PlayheadLowSettings)
    playhead_flash:     Group[PlayheadFlashSettings]    = Group(PlayheadFlashSettings)
    playhead_haunted:   Group[PlayheadHauntedSettings]  = Group(PlayheadHauntedSettings)
    playhead_test:      Group[PlayheadTestSettings]     = Group(PlayheadTestSettings)


class HighLayersSettings(BaseSettings):
    """Per-layer composition settings — the high (ring-regime) block of the pool, plus
    `wind_down` (cross-regime, classed LowLayer) sitting next to flood, its ending.
    `fov` is a hidden relay (from the root) into test_player_lines/test_calibration."""
    fov: Field[float] = Field(110.0, min=60.0, max=180.0, step=0.5, visible=False, description="Camera horizontal FOV — hidden relay to test_player_lines/test_calibration")
    pose_instrument:    Group[PoseInstrumentSettings]  = Group(PoseInstrumentSettings)
    playhead_high:      Group[PlayheadHighSettings]    = Group(PlayheadHighSettings)
    flood:              Group[FloodSettings]           = Group(FloodSettings)
    wind_down:          Group[WindDownSettings]        = Group(WindDownSettings)
    test_pose_waves:    Group[PoseWavesSettings]       = Group(PoseWavesSettings)
    test_harmonic:      Group[HarmonicSettings]        = Group(HarmonicSettings)
    test_player_lines:  Group[PlayerLinesSettings]     = Group(PlayerLinesSettings, share=[fov.as_('fov')])
    test_calibration:   Group[CameraLightSettings]     = Group(CameraLightSettings, share=[fov.as_('fov')])
    test_fill:          Group[FillSettings]            = Group(FillSettings)
    test_pulse:         Group[PulseSettings]           = Group(PulseSettings)
    test_chase:         Group[ChaseSettings]           = Group(ChaseSettings)
    test_lines:         Group[LinesSettings]           = Group(LinesSettings)
    test_random:        Group[RandomSettings]          = Group(RandomSettings)


class LightSettings(BaseSettings):
    """Settings for the LED light system (conductor thread + layers + compositor)."""

    # Construction / wiring (INIT — requires restart to take effect)
    max_poses:        Field[int]   = Field(3,    min=1,   max=16,   access=Field.INIT, description="Max tracked poses")
    num_cameras:      Field[int]   = Field(1,    min=1,   max=16,   access=Field.INIT, description="Number of cameras")
    light_rate:       Field[float] = Field(30.0, min=1,   max=120,  access=Field.INIT, description="Light output frame rate (fps)")
    light_resolution: Field[int]   = Field(3600, min=256, max=4000, access=Field.INIT, description="LED strip resolution (pixels)")
    fov: Field[float] = Field(110.0, min=60.0, max=180.0, step=0.5, description="Camera horizontal FOV — hidden relay from root to player_lines/calibration")

    master:         Field[float] = Field(1.0, min=0.0, max=1.0, step=0.01, description="Master brightness (applied to the composite; lamp gamma/floor live in the light sender)", newline=True)
    light_phase: Field[float]    = Field(0.0, min=0.0, max=1.0, step=0.01, description="High-speed ring offset (0–1 turn), applied to the spun-content layers")

    # Debug override — a first-class select ABOVE the state machine: choosing a layer IS
    # turning debug on. The Compositor draws that one layer solo (full weight) and the
    # motor auto-follows its regime (HighLayer → HIGH, LowLayer → LOW); OFF returns the
    # show where it would have been. Forced OFF at startup (boot failsafe: a preset saved
    # mid-debug must never spin at power-on).
    debug: Field[DebugLayer] = Field(DebugLayer.OFF, description="Debug override: select a layer to show it solo and auto-follow the motor to its regime (OFF = show runs)", newline=True)

    clock:        Group[ClockSettings]        = Group(ClockSettings)
    motor:        Group[MotorSettings]        = Group(MotorSettings)
    playhead:     Group[PlayheadSettings]     = Group(PlayheadSettings)

    # The per-layer composition settings pool, grouped by regime (mirrors layers/low|high/).
    low_layers:  Group[LowLayersSettings]  = Group(LowLayersSettings)
    high_layers: Group[HighLayersSettings] = Group(HighLayersSettings, share=[fov.as_('fov')])
