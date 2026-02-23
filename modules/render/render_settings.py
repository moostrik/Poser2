"""RenderSettings — reactive settings for the render pipeline.

Manages mutable runtime state (feature selection, data layer mode, stages,
LUT, flow/fluid configs) via the BaseSettings descriptor system. Registered
in the SettingsRegistry for NiceGUI panel and JSON preset persistence.

Window/init fields (title, width, fps, …) are init_only Settings.
"""

from enum import IntEnum, auto
from typing import Protocol

from modules.DataHub import Stage
from modules.pose.Frame import ScalarFrameField
from modules.render.layers.generic.CompositeLayer import LutSelection
from modules.render.layers.flow.FlowLayer import FlowLayerConfig
from modules.render.layers.flow.FluidLayer import FluidLayerConfig

from modules.settings import Setting, Child, BaseSettings


# ---------------------------------------------------------------------------
# Enums & protocols
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Settings classes
# ---------------------------------------------------------------------------

class WindowSettings(BaseSettings):
    """Window / init configuration — set once before RenderManager starts."""
    title: Setting[str] =           Setting(str,  "Poser", init_only=True, visible=False)
    fps: Setting[int] =             Setting(int,  60, readonly=True)
    v_sync: Setting[bool] =         Setting(bool, True)
    fullscreen: Setting[bool] =     Setting(bool, False)
    monitor: Setting[int] =         Setting(int,  0)
    secondary_list: Setting[list[int]] = Setting(list[int], [])
    x: Setting[int] =               Setting(int,  0)
    y: Setting[int] =               Setting(int,  80)
    width: Setting[int] =           Setting(int,  1920)
    height: Setting[int] =          Setting(int,  1000)


class DataLayerSettings(BaseSettings):
    """Data layer control — feature selection, display mode, stages."""
    feature: Setting[RenderFeature] =   Setting(RenderFeature, RenderFeature.angle_motion)
    mode: Setting[LayerMode] =      Setting(LayerMode, LayerMode.WINDOW)
    stage_a: Setting[Stage] =       Setting(Stage, Stage.SMOOTH)
    stage_b: Setting[Stage] =       Setting(Stage, Stage.LERP)


class RenderSettings(BaseSettings):
    """Mutable render-pipeline settings (registered in SettingsRegistry)."""

    # Window / init-only (foldable child)
    window: Child[WindowSettings] =     Child(WindowSettings)

    # Data layer control (foldable child)
    data_layer: Child[DataLayerSettings] =  Child(DataLayerSettings)

    # LUT
    lut: Setting[LutSelection] =           Setting(LutSelection, LutSelection.NONE)  # type: ignore[attr-defined]
    lut_strength: Setting[float] =  Setting(float, 1.0, min=0.0, max=1.0)

    # Flow / Fluid (child configs, shared across all cameras)
    flow: Child[FlowLayerConfig] =          Child(FlowLayerConfig)
    fluid: Child[FluidLayerConfig] =         Child(FluidLayerConfig)

    # ------------------------------------------------------------------
    # Data-layer binding (replaces Config.bind / _propagate)
    # ------------------------------------------------------------------

    def bind_data_layers(self,
             windows_a: dict[int, DataLayer], frames_a: dict[int, DataLayer], angvel_a: dict[int, DataLayer],
             windows_b: dict[int, DataLayer], frames_b: dict[int, DataLayer], angvel_b: dict[int, DataLayer]) -> None:
        """Bind data-layer instances so config changes propagate active state."""
        object.__setattr__(self, '_slots', [
            (windows_a, frames_a, angvel_a, 'stage_a'),
            (windows_b, frames_b, angvel_b, 'stage_b'),
        ])
        # Wire _propagate to each relevant field on the data_layer child
        dl = self.data_layer
        for field in (DataLayerSettings.feature, DataLayerSettings.mode, DataLayerSettings.stage_a, DataLayerSettings.stage_b):
            dl.bind(field, lambda _v: self._propagate())
        self._propagate()

    def _propagate(self) -> None:
        """Push current config to bound data layers."""
        dl = self.data_layer
        ff = dl.feature
        scalar_field = None if ff == RenderFeature.NONE else ScalarFrameField(ff.value)
        is_angles = (scalar_field == ScalarFrameField.angles) if scalar_field else False

        for windows, frames, angvel, stage_attr in self._slots:
            stage: Stage = getattr(dl, stage_attr)
            for i in windows:
                # Deactivate all three layer types
                windows[i].set_active(False)
                frames[i].set_active(False)
                angvel[i].set_active(False)

                # Activate and configure the selected layer
                if scalar_field is not None:
                    layer = windows[i] if dl.mode == LayerMode.WINDOW else (angvel[i] if is_angles else frames[i])
                    layer.set_active(True)

                    cfg = layer._config
                    cfg.feature_field = scalar_field
                    cfg.stage = stage
