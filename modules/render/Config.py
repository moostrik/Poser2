from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Protocol

from modules.ConfigBase import ConfigBase, config_field
from modules.DataHub import Stage
from modules.pose.Frame import FrameField, ScalarFrameField


# Auto-generated list of scalar features from Frame definition
DATA_FEATURES: list[FrameField] = FrameField.get_scalar_fields()


class RenderFeature(IntEnum):
    """Feature field selection for rendering. NONE disables the feature."""
    NONE =          -1
    bbox =          ScalarFrameField.bbox
    angles =        ScalarFrameField.angles
    angle_vel =     ScalarFrameField.angle_vel
    angle_motion =  ScalarFrameField.angle_motion
    angle_sym =     ScalarFrameField.angle_sym
    similarity =    ScalarFrameField.similarity
    leader =        ScalarFrameField.leader
    motion_gate =   ScalarFrameField.motion_gate


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
    title: str =         config_field("Poser",   fixed=True)
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
    feature: RenderFeature = config_field(RenderFeature.angle_motion)
    mode: LayerMode =        config_field(LayerMode.WINDOW)
    A: Stage =          config_field(Stage.SMOOTH)
    B: Stage =          config_field(Stage.LERP)

    def bind(self,
             windows_a: dict[int, DataLayer], frames_a: dict[int, DataLayer], angvel_a: dict[int, DataLayer],
             windows_b: dict[int, DataLayer], frames_b: dict[int, DataLayer], angvel_b: dict[int, DataLayer]) -> None:
        """Bind layer instances to this config. Config changes propagate active state to layers."""
        self._slots = [
            (windows_a, frames_a, angvel_a, 'A'),
            (windows_b, frames_b, angvel_b, 'B'),
        ]
        self.watch(self._propagate)
        self._propagate()

    def _propagate(self) -> None:
        """Push current config to bound layers, setting active state and shared properties."""
        ff = self.feature
        scalar_field = None if ff == RenderFeature.NONE else ScalarFrameField(ff.value)
        is_angles = (scalar_field == ScalarFrameField.angles) if scalar_field else False

        for windows, frames, angvel, stage_attr in self._slots:
            stage: Stage = getattr(self, stage_attr)
            for i in windows:
                # Deactivate all three layer types
                windows[i].set_active(False)
                frames[i].set_active(False)
                angvel[i].set_active(False)

                # Activate and configure the selected layer if feature is not NONE
                if scalar_field is not None:
                    layer = windows[i] if self.mode == LayerMode.WINDOW else (angvel[i] if is_angles else frames[i])
                    layer.set_active(True)

                    # Update shared config properties (all layers see these changes)
                    cfg = layer._config
                    cfg.feature_field = scalar_field
                    cfg.stage = stage