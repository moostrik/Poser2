from enum import IntEnum, auto
from modules.settings import BaseSettings, Field, Group, Widget


from .clock import ClockSettings
from .motor import MotorSettings
from .playhead import PlayheadSettings
from .layers import (
    PoseWavesSettings, FillSettings, PulseSettings,
    ChaseSettings, LinesSettings, RandomSettings, HarmonicSettings,
    PlayerLinesSettings, CameraLightSettings, PlayheadFlashSettings,
    PlayheadLowSettings, PlayheadHighSettings,
)


class LowLayerId(IntEnum):
    """idle/low slot options: the low (slow) layers + the shared test layers (empty = black)."""
    playhead       = auto()
    playhead_flash = auto()
    fill           = auto()
    pulse          = auto()
    chase          = auto()
    lines          = auto()
    random         = auto()


class HighLayerId(IntEnum):
    """high slot options: the high layers + the shared test layers (empty = black)."""
    pose_waves   = auto()
    harmonic     = auto()
    player_lines = auto()
    calibration  = auto()
    playhead     = auto()
    fill         = auto()
    pulse        = auto()
    chase        = auto()
    lines        = auto()
    random       = auto()


class LowCompSettings(BaseSettings):
    """Low (slow) composition settings."""
    playhead:       Group[PlayheadLowSettings]     = Group(PlayheadLowSettings)
    playhead_flash: Group[PlayheadFlashSettings]   = Group(PlayheadFlashSettings)


class HighCompSettings(BaseSettings):
    """High composition settings. `fov` is a hidden relay (from the root) into player_lines/calibration."""
    fov: Field[float] = Field(110.0, min=60.0, max=180.0, step=0.5, visible=False, description="Camera horizontal FOV — hidden relay to player_lines/calibration")
    pose_waves:   Group[PoseWavesSettings]    = Group(PoseWavesSettings)
    harmonic:     Group[HarmonicSettings]     = Group(HarmonicSettings)
    player_lines: Group[PlayerLinesSettings]  = Group(PlayerLinesSettings,  share=[fov.as_('fov')])
    calibration:  Group[CameraLightSettings]  = Group(CameraLightSettings,   share=[fov.as_('fov')])
    playhead:     Group[PlayheadHighSettings] = Group(PlayheadHighSettings)


class TestCompSettings(BaseSettings):
    """Test composition settings — generic patterns shared by the low and high slots."""
    fill:   Group[FillSettings]   = Group(FillSettings)
    pulse:  Group[PulseSettings]  = Group(PulseSettings)
    chase:  Group[ChaseSettings]  = Group(ChaseSettings)
    lines:  Group[LinesSettings]  = Group(LinesSettings)
    random: Group[RandomSettings] = Group(RandomSettings)


class LightSettings(BaseSettings):
    """Settings for the LED light renderer thread."""

    # Construction / wiring (INIT — requires restart to take effect)
    max_poses:        Field[int]   = Field(3,    min=1,   max=16,   access=Field.INIT, description="Max tracked poses")
    num_cameras:      Field[int]   = Field(1,    min=1,   max=16,   access=Field.INIT, visible=False, description="Number of cameras")
    light_rate:       Field[float] = Field(30.0, min=1,   max=120,  access=Field.INIT, description="Light output frame rate (fps)")
    light_resolution: Field[int]   = Field(3600, min=256, max=4000, access=Field.INIT, visible=False, description="LED strip resolution (pixels)")
    fov: Field[float] = Field(110.0, min=60.0, max=180.0, step=0.5, visible=False, description="Camera horizontal FOV — hidden relay from root to player_lines/calibration")

    # Per-motor-speed layer selectors (multi-select; layers in a slot blend, slots crossfade by rpm)
    idle_layers:  Field[list[LowLayerId]]  = Field([LowLayerId.playhead], widget=Widget.checklist, description="Layers shown at idle", newline=True)
    low_layers:   Field[list[LowLayerId]]  = Field([LowLayerId.playhead_flash],widget=Widget.checklist, description="Layers shown at low speed")
    high_layers:  Field[list[HighLayerId]] = Field([HighLayerId.pose_waves],   widget=Widget.checklist, description="Layers shown at high speed (light_phase applied)")

    master:         Field[float] = Field(1.0, min=0.0, max=1.0, step=0.01, description="Master brightness (applied to the composite; lamp gamma/floor live in osc_light)", newline=True)
    high_cross_rpm: Field[float] = Field(1200.0, min=0.0, max=2400.0, step=10.0, description="LOW→HIGH crossfade rpm")
    light_phase: Field[float]    = Field(0.0, min=0.0, max=1.0, step=0.01, description="High-slot ring offset (0–1 turn)")

    clock:        Group[ClockSettings]        = Group(ClockSettings)
    motor:        Group[MotorSettings]        = Group(MotorSettings)
    playhead:     Group[PlayheadSettings]     = Group(PlayheadSettings)

    # Composition settings grouped by category (mirrors the layers/ low|high|test subpackages)
    low:  Group[LowCompSettings]  = Group(LowCompSettings)
    high: Group[HighCompSettings] = Group(HighCompSettings, share=[fov.as_('fov')])   # relay fov into high
    test: Group[TestCompSettings] = Group(TestCompSettings)
