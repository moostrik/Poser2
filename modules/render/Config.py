from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Protocol

from modules.ConfigBase import ConfigBase, config_field
from modules.DataHub import Stage
from modules.pose.Frame import FrameField, ScalarFrameField


# Auto-generated list of scalar features from Frame definition
DATA_FEATURES: list[FrameField] = FrameField.get_scalar_fields()


class SlotStage(IntEnum):
    """Stage selection for data layer slots. NONE disables the slot."""
    NONE =      -1
    RAW =       Stage.RAW
    SMOOTH =    Stage.SMOOTH
    LERP =      Stage.LERP


class LayerMode(IntEnum):
    """Display mode for a data layer slot."""
    FRAME =     0
    WINDOW =    auto()


class DataLayer(Protocol):
    """Protocol for data layer instances with active state and shared config."""
    _config: 'DataLayerConfig'

    def set_active(self, active: bool) -> None: ...

class DataLayerConfig(Protocol):
    """Protocol for data layer shared configs."""
    feature_field: ScalarFrameField
    stage: Stage


@dataclass
class Config(ConfigBase):
    """Render configuration with data layer binding."""
    title: str =         config_field("Poser",  fixed=True)
    monitor: int =       config_field(0,         fixed=True)
    width: int =         config_field(1920,      fixed=True)
    height: int =        config_field(1000,      fixed=True)
    x: int =             config_field(0,         fixed=True)
    y: int =             config_field(80,        fixed=True)
    fullscreen: bool =   config_field(False,     fixed=True)
    fps: int =           config_field(60,        fixed=True)
    v_sync: bool =       config_field(True,      fixed=True)

    secondary_list: list[int] = config_field(default_factory=list, fixed=True)  # type: ignore[assignment]

    num_cams: int =      config_field(3,         repr=False)
    num_players: int =   config_field(3,         repr=False)

    cams_a_row: int =    config_field(3,         fixed=True)
    num_R: int =         config_field(3,         fixed=True)
    stream_capacity: int = config_field(600,     fixed=True, description="10 seconds at 60 FPS")

    # Data layer config (mutable — runtime switching via GUI / watch)
    feature: ScalarFrameField = config_field(ScalarFrameField.angle_motion)
    mode: LayerMode =           config_field(LayerMode.WINDOW)
    stage_a: SlotStage =        config_field(SlotStage.SMOOTH)
    stage_b: SlotStage =        config_field(SlotStage.LERP)

    def bind(self,
             windows_a: dict[int, DataLayer], frames_a: dict[int, DataLayer], angvel_a: dict[int, DataLayer],
             windows_b: dict[int, DataLayer], frames_b: dict[int, DataLayer], angvel_b: dict[int, DataLayer]) -> None:
        """Bind layer instances to this config. Config changes propagate active state to layers."""
        self._slots = [
            (windows_a, frames_a, angvel_a, 'stage_a'),
            (windows_b, frames_b, angvel_b, 'stage_b'),
        ]
        self.watch(self._propagate)
        self._propagate()

    def _propagate(self) -> None:
        """Push current config to bound layers, setting active state and shared properties."""
        ff = self.feature
        is_angles = (ff == ScalarFrameField.angles)

        for windows, frames, angvel, stage_attr in self._slots:
            slot_stage: SlotStage = getattr(self, stage_attr)
            for i in windows:
                # Deactivate all three layer types
                windows[i].set_active(False)
                frames[i].set_active(False)
                angvel[i].set_active(False)

                # Activate and configure the selected layer
                if slot_stage != SlotStage.NONE:
                    layer = windows[i] if self.mode == LayerMode.WINDOW else (angvel[i] if is_angles else frames[i])
                    layer.set_active(True)

                    # Update shared config properties (all layers see these changes)
                    cfg = layer._config
                    cfg.feature_field = ff
                    cfg.stage = Stage(slot_stage)