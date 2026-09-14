from enum import IntEnum, auto
from modules.settings import BaseSettings, Field, Group


from .clock import ClockSettings
from .motor import MotorSettings
from .playhead import PlayheadSettings
from .layers import (
    BeamBlueSoundSettings, BeamPlayheadSettings, BeamFlashSettings, BeamWindDownSettings,
    BeamHauntedSettings, BeamTestSettings,
    PoseInstrumentSettings, ProjectionPlayheadSettings, FloodSettings,
    TestPlayerLinesSettings, TestCalibrationSettings, TestFillSettings, TestPulseSettings,
    TestChaseSettings, TestLinesSettings, TestRandomSettings, TestPoseWavesSettings,
    TestHarmonicSettings,
)


class LayerId(IntEnum):
    """The unified layer pool: everything a state (or the debug select) can put in a mix.

    One instance per layer; each layer's mode (beam = lamps, projection = pixels) lives in its
    class (`BeamLayer`/`ProjectionLayer`). Member order is the beam block then the projection
    block — it drives the `DebugLayer` dropdown order and mirrors the settings groups.
    Debug-only layers (never in a state's mix) state that role in their docstrings; the
    projection block's generic patterns keep the ``test_`` prefix.
    """
    # beam mode — the lamps
    beam_blue_sound     = auto()   # beam: soundscape levels on the left/right blue lamps
    beam_playhead       = auto()   # beam: front white lamp (the searchlight line)
    beam_flash          = auto()   # beam: flash as the playhead crosses a participant
    beam_wind_down      = auto()   # beam: flood's ending — the two white lamps fading over the spin-down (the wall while fast)
    beam_haunted        = auto()   # beam: player/ghost flash (debug/experimentation)
    beam_test           = auto()   # beam: direct levels for the four physical lamps (debug)
    # projection mode — the projection
    pose_instrument     = auto()   # projection: the pose instrument — each person's pose-patched line pattern (see LAYERS.md)
    projection_playhead = auto()   # projection: bright marker visualising the content playhead
    flood               = auto()   # projection: the whole projection constant white (S8's wall)
    test_player_lines   = auto()
    test_calibration    = auto()
    test_fill           = auto()
    test_pulse          = auto()
    test_chase          = auto()
    test_lines          = auto()
    test_random         = auto()
    test_pose_waves     = auto()   # projection: the old wave/void instrument (reference/montage)
    test_harmonic       = auto()


class DebugLayer(IntEnum):
    """The debug select: OFF = debug disarmed (the show runs); any other member = debug on,
    showing exactly that layer solo while the motor auto-follows its mode.

    Mirrors `LayerId` member for member (same names, same values, same order) with OFF = 0
    in front. Spelled out rather than generated so type checkers see a real enum;
    `tests/test_debug_layer.py` fails if the two ever drift apart.
    """
    OFF                 = 0
    # beam mode — the lamps
    beam_blue_sound     = auto()
    beam_playhead       = auto()
    beam_flash          = auto()
    beam_wind_down      = auto()
    beam_haunted        = auto()
    beam_test           = auto()
    # projection mode — the projection
    pose_instrument     = auto()
    projection_playhead = auto()
    flood               = auto()
    test_player_lines   = auto()
    test_calibration    = auto()
    test_fill           = auto()
    test_pulse          = auto()
    test_chase          = auto()
    test_lines          = auto()
    test_random         = auto()
    test_pose_waves     = auto()
    test_harmonic       = auto()


class BeamLayersSettings(BaseSettings):
    """Per-layer composition settings — the beam-mode block of the pool.
    `spin_down_seconds` is a hidden relay (from the root) into wind_down — the spin-down
    slider's visible home is the states panel, next to spin_up."""
    spin_down_seconds: Field[float] = Field(10.0, min=1.0, max=60.0, step=0.5, visible=False, description="S9/S10 wall-fade seconds — hidden relay from states (via the root) into wind_down")
    beam_blue_sound:    Group[BeamBlueSoundSettings]    = Group(BeamBlueSoundSettings)
    beam_playhead:      Group[BeamPlayheadSettings]     = Group(BeamPlayheadSettings)
    beam_flash:         Group[BeamFlashSettings]        = Group(BeamFlashSettings)
    beam_wind_down:     Group[BeamWindDownSettings]     = Group(BeamWindDownSettings, share=[spin_down_seconds.as_('spin_down_seconds')])
    beam_haunted:       Group[BeamHauntedSettings]      = Group(BeamHauntedSettings)
    beam_test:          Group[BeamTestSettings]         = Group(BeamTestSettings)


class ProjectionLayersSettings(BaseSettings):
    """Per-layer composition settings — the projection-mode block of the pool.
    `fov` is a hidden relay (from the root) into the calibration layer."""
    fov: Field[float] = Field(110.0, access=Field.INIT, description="Camera horizontal FOV — relay to test_calibration")
    pose_instrument:    Group[PoseInstrumentSettings]        = Group(PoseInstrumentSettings)
    projection_playhead: Group[ProjectionPlayheadSettings]   = Group(ProjectionPlayheadSettings)
    flood:              Group[FloodSettings]           = Group(FloodSettings)
    test_player_lines:  Group[TestPlayerLinesSettings] = Group(TestPlayerLinesSettings)
    test_calibration:   Group[TestCalibrationSettings] = Group(TestCalibrationSettings, share=[fov.as_('fov')])
    test_fill:          Group[TestFillSettings]        = Group(TestFillSettings)
    test_pulse:         Group[TestPulseSettings]       = Group(TestPulseSettings)
    test_chase:         Group[TestChaseSettings]       = Group(TestChaseSettings)
    test_lines:         Group[TestLinesSettings]       = Group(TestLinesSettings)
    test_random:        Group[TestRandomSettings]      = Group(TestRandomSettings)
    test_pose_waves:    Group[TestPoseWavesSettings]   = Group(TestPoseWavesSettings)
    test_harmonic:      Group[TestHarmonicSettings]    = Group(TestHarmonicSettings)


class LightSettings(BaseSettings):
    """Settings for the LED light system (conductor thread + layers + compositor)."""

    # Construction / wiring (INIT — requires restart to take effect)
    max_poses:        Field[int]   = Field(3,    min=1,   max=16,   access=Field.INIT, description="Max tracked poses")
    num_cameras:      Field[int]   = Field(1,    min=1,   max=16,   access=Field.INIT, description="Number of cameras")
    light_rate:       Field[float] = Field(30.0, min=1,   max=120,  access=Field.INIT, description="Light output frame rate (fps)")
    light_resolution: Field[int]   = Field(3600, min=256, max=4000, access=Field.INIT, description="Projection resolution (pixels per turn)")
    fov: Field[float] = Field(110.0, access=Field.INIT, description="Camera horizontal FOV — relay from root to test_calibration; its visible home is the camera panel")
    spin_down_seconds: Field[float] = Field(10.0, min=1.0, max=60.0, step=0.5, visible=False, description="S9/S10 wall-fade seconds — hidden relay from states (via the root) into wind_down")

    brightness:     Field[float] = Field(1.0, min=0.0, max=1.0, step=0.01, description="Main brightness — applied to the composite; the lamp gamma and floor live in the light sender", newline=True, pinned=True)

    # Debug override — a first-class select ABOVE the state machine: choosing a layer IS
    # turning debug on. The Compositor draws that one layer solo (full weight) and the
    # motor auto-follows its mode (ProjectionLayer → PROJECTION, BeamLayer → BEAM); OFF returns the
    # show where it would have been. Forced OFF at startup (boot failsafe: a preset saved
    # mid-debug must never spin at power-on).
    debug: Field[DebugLayer] = Field(DebugLayer.OFF, description="Debug override: select a layer to show it solo and auto-follow the motor to its mode (OFF = show runs)")

    clock:        Group[ClockSettings]        = Group(ClockSettings)
    motor:        Group[MotorSettings]        = Group(MotorSettings)
    playhead:     Group[PlayheadSettings]     = Group(PlayheadSettings)

    # The per-layer composition settings pool, grouped by mode (mirrors layers/beam|projection/).
    beam_layers:       Group[BeamLayersSettings]       = Group(BeamLayersSettings, share=[spin_down_seconds.as_('spin_down_seconds')])
    projection_layers: Group[ProjectionLayersSettings] = Group(ProjectionLayersSettings, share=[fov.as_('fov')])
